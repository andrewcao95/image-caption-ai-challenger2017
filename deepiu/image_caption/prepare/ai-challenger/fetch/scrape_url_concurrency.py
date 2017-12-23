#!/usr/bin/env python
#-*- coding:gbk -*-

import os
import sys
import fileinput
import subprocess
import urllib
import json
import time
import httplib
import Queue
import threading

query_file_path = 'query.txt'
output_file_path = 'query.json'
error_file_path = 'query.err.txt'
parsed_file_path = 'query.res'

rank_num=120

black_query_file_path = ''
black_url_file_path = ''

black_queries = {}
black_urls = {}

"""
Load black query list
"""
def load_black_query_list(filepath):
    global black_queries
    print 'Begin to load black query list:', filepath
    # TODO add load logic
    print 'End to load black query list:', filepath
    return 0

def load_black_url_list(filepath):
    global black_urls
    print 'Begin to load black url list:', filepath
    # TODO add load logic
    print 'End to load black url list:', filepath
    return 0;

class Worker(threading.Thread):
    def __init__(self, workQueue, resultQueue, timeout=1, **kwargs):
        threading.Thread.__init__(self, kwargs=kwargs)
        self.timeout = timeout
        self.workQueue = workQueue
        self.resultQueue = resultQueue
        self.end = False

    def stop(self):
        self.end = True


    def run(self):
        while True:
            try:
                callable, args, kwargs = self.workQueue.get(timeout=self.timeout)
                res = callable(*args, **kwargs)
                self.resultQueue.put(res)
            except Queue.Empty:
                if self.end:
                    print "End thread", self
                    break
            except :
                print sys.exc_info()
                raise

class ControledThreadPool:
    def __init__(self, num_of_threads=10, max_qps=10):
        self.workQueue = Queue.Queue(max_qps)
        self.resultQueue = Queue.Queue(max_qps)
        self.max_qps = max_qps
        self.workers = []
        self.__create_thread_pool(num_of_threads)
        self.job_counter = 0
        self.start_time = time.time()

    def __create_thread_pool(self, num_of_threads):
        for i in range(num_of_threads):
            thread = Worker(self.workQueue, self.resultQueue)
            self.workers.append(thread)

    def wait_for_complete(self):
        for work in self.workers:
            work.stop()
        while len(self.workers) > 0:
            work = self.workers.pop()
            if work.isAlive():
                work.join()

    def run(self):
        for work in self.workers:
            work.start()

    def add_job(self, callable, *args, **kwargs):
        # qps control
        self.job_counter += 1
        if self.job_counter % self.max_qps == 0:
            end_time = time.time()
            if end_time-self.start_time < 1.0:
                print 'sleep', self.start_time+1.0-end_time
                time.sleep(self.start_time+1.0-end_time)
            self.start_time = time.time()

        self.workQueue.put((callable, args, kwargs))

    def get_result(self):
        return self.resultQueue.get()

"""
scrape image data of query and return result
"""
def scrape_query_result(query,pn,rn):
    result = (query, '')
    sUserCode = 'iu_adimage'
    # decode from gbk to unicode ,than encode to utf8, because img server accepts utf8
    try:
        uquery = query.decode('gbk')
        uquery = uquery.encode('utf8')
        params = {'word':uquery, 'pn':pn, 'rn':rn, 'user':sUserCode}
        conn=httplib.HTTPConnection('imgdata.baidu.com')
        conn.request('GET', '/platform/search?'+urllib.urlencode(params))
        response = conn.getresponse()
        if response.status != 200:
            print time.strftime('%Y-%m-%d %H:%M:%S'), query,'fail', response.status, 'errorcode='+str(response.status), 'reason='+response.read().strip()
        else:
            result = (query, response.read().strip())
            print time.strftime('%Y-%m-%d %H:%M:%S'), query, 'success'
        return result
    except:
        return result

class ResultWriterThread(threading.Thread):
    def __init__(self, output_file, error_file, pool, **kwargs):
        threading.Thread.__init__(self, kwargs=kwargs)
        self.output_file = output_file
        self.error_file = error_file
        self.pool = pool

    def run(self):
        while True:
            (query, image_data) = self.pool.get_result()
            if image_data == '':
                self.error_file.write(query+'\n')
            else:
                self.output_file.write(query+'\t'+image_data+'\n')

class QueryReaderThread(threading.Thread):
    def __init__(self, query_file, black_queries, pool, **kwargs):
        threading.Thread.__init__(self, kwargs=kwargs)
        self.filepath = query_file
        self.pool = pool
        self.black_queries = black_queries

    def run(self):
        for line in fileinput.input(self.filepath):
            line = line.strip();
            if line == "":
                continue
            items=line.strip().split('\t')
            query=items[0]
            if len(query) == 0 or query in black_queries:
                continue
            # add scrape job to pool
            i=0
            while i < rank_num:
                self.pool.add_job(scrape_query_result, query, i ,60)
                i=i+60


"""
load query file and get image data
"""
def get_image_data(filepath, output_filepath, error_file):
    global black_queries
    sQPS = 10
    output_file = file(output_filepath, 'w')
    request_count = 0;
    start_time = time.time()
    pool = ControledThreadPool(max_qps = sQPS)
    reader = QueryReaderThread(filepath, black_queries, pool)
    writer = ResultWriterThread(output_file, error_file, pool)
    writer.setDaemon(True)
    reader.start()
    writer.start()
    pool.run()
    print 'begin reader.jion'
    sys.stdout.flush()
    reader.join()
    print 'end reader.jion'
    sys.stdout.flush()
    pool.wait_for_complete()
    print 'end pool'
    sys.stdout.flush()
    output_file.close();
    return 0;

"""
Parse image data and print to new file
Sample: query    {
"status":{"code": "0","msg": "success" },
"data": {"TotalNumber":1000,"ReturnNumber":1,"ResultArray":
        [{"Key":"3889257040,4098799047",
        "ObjUrl":"http:\/\/www.logosnews.tw\/wp-content\/uploads\/2013\/12\/Game11-banner.jpg",
        "FromUrl":"http:\/\/www.logosnews.tw\/?p=7896",
        "Width":750,"Height":400,
        "Desc":"<strong>\u6211\u662f\u8c01<\/strong>?","Pictype":"jpg",
        "ThumbnailUrl":"http:\/\/img3.imgtn.bdimg.com\/it\/u=3889257040,4098799047&fm=21&gp=0.jpg",
        "Pagenum":0,"Di":199875369320,"MidUrl":"","LargeUrl":"",
        "ThumWidth":413,"ThumHeight":220,"MidWidth":0,"MidHeight":0,
        "LargeWidth":0,"LargeHeight":0,"IsSet":0,"SetObjNum":0,
        "SetId":"0,0","SetArray":[]}],
        "Column":""}}

output file format: field in [] is option
query    ThumbnailUrl=xxx    ObjUrl=xxx    FromUrl=xxx    Desc=xxx    ThumSize=width*height    ObjSize=width*height    [ObjLength=xxx]    [Pictype=xxx]    [ObjSign=xxx]
"""
def parse_image_data(filepath, parsed_filepath, error_file):
    parsed_file = file(parsed_filepath,'w')
    sOutputImageFields = {
            'Pagenum':'Pagenum',
            'ThumbnailUrl':'ThumbnailUrl',
            'Key':'Key',
            'ObjUrl':'ObjUrl',
            #'FromUrl':'FromUrl',
            #'Desc':'Desc',
            #'ThumSize':'ThumWidth*ThumHeight',
            #'ObjSize':'Width*Height',
            #'ObjLength':'ObjLength',
            'Pictype':'Pictype'
            #, 'ObjSign':'ObjSign'
            }
    line_index = 0;
    for line in fileinput.input(filepath):
        line_index+=1
        line = line.strip()
        fields = line.split('\t', 2)
        if len(line) == 0:
            continue
        if (len(fields)!=2 or len(fields[0])==0 or len(fields[1]) == 0):
            print fields[0],'fail format', line
            continue
        image_obj = json.loads(fields[1])
        if image_obj['status']['code'] != '0' or image_obj['status']['msg']!='success':
            print fields[0], 'fail status', fields[1]
            error_file.write(fields[0]+'\n')
            continue
        image_index = 0
        for image in image_obj['data']['ResultArray']:
            image_index+=1
            #parsed_file.write(fields[0].encode('gbk'))
            parsed_file.write(fields[0])
            for key, value_name in sOutputImageFields.items():
                value = ''
                mult_value = value_name.split('*')
                if len(mult_value) == 1:
                    if value_name in image:
                        value = image[value_name]
                elif len(mult_value) == 2:
                    if mult_value[0] in image and mult_value[1] in image:
                        value = '%s*%s'%(image[mult_value[0]],
                            image[mult_value[1]])
                #print key,value
                if value == '' or value is None:
                    print fields[0], line_index, image_index, key,value
                    parsed_file.write('\t%s=%s'%(key,value))
                else:
                    try:
                        if key.find("Pagenum")>-1:
                            parsed_file.write('\t%s=%d'%(key, value))
                        else:
                            parsed_file.write('\t%s=%s'%(key, value.encode('gbk')))
                    except UnicodeEncodeError, e:
                        print fields[0], line_index, image_index, key, 'UnicodeEncodeError'
            parsed_file.write('\n')
    parsed_file.close()
    return 0



def main():
    global black_query_file_path
    global black_url_file_path
    global query_file_path
    global output_file_path
    global error_file_path
    global parsed_file_path
    ret = load_black_query_list(black_query_file_path)
    if ret != 0:
        return ret
    ret = load_black_url_list(black_url_file_path)
    if ret != 0:
        return ret

    error_file = file(error_file_path, 'w')
    #TODO merge get_image_data and parse_image_data
    ret = get_image_data(query_file_path, output_file_path, error_file)
    if ret != 0:
        error_file.close()
        return ret
    ret = parse_image_data(output_file_path, parsed_file_path, error_file)
    if ret != 0:
        error_file.close()
        return ret
    error_file.close()

    # TODO 1.store to gips, need to split image to 100k one time runing
    # TODO 2.merge image_data and gips result
    return 0


if __name__ == '__main__':
    start = time.time()
    ret = main()
    end = time.time()
    print 'Total run time', end-start, 'seconds'
    if (ret != 0):
        print 'return', ret, 'total run time', end-start, 'seconds'
        sys.exit(1)
    else:
        print 'return', ret, 'total run time', end-start, 'seconds'
        sys.exit(0)
