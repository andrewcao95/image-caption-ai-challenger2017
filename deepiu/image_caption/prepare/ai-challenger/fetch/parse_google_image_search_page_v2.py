#!/usr/bin/env python
#encoding:gbk

import sys,os
from bs4 import BeautifulSoup
from bs4 import element
import traceback
import json

reload(sys)
sys.setdefaultencoding("gbk")

def load_index_kw(filename):
	index_kw = {}
	f = open(filename, "r")
	for line in f:
		line = line.strip("\r\n")
		fs = line.split("\t")
		index_kw[fs[-1]] = "\t".join(fs[0:-1])
	f.close()
	return index_kw

#index_kw = load_index_kw("google_image_query_index")

def read_raw_page(html_filename, try_num=3, *argv):
	contents_str = ""
	for i in xrange(try_num):
		try:	
			content = open(html_filename, "r")
			for line in content:
				contents_str += line
			#contents_str = contents_str.encode("utf-8")
			content.close()
			break
		except Exception, e:
			if i == try_num-1:
				print >>sys.stderr, "read_failed_file:%s" % url
				print >>sys.stderr, traceback.print_exc() 
				print >>sys.stderr, "===="
			else:
				continue
	return contents_str

def parse_google_pages(kw, html_filename):
	contents_str = read_raw_page(html_filename)
	result = []

	if contents_str:
		result = parse_result(contents_str, html_filename)
	for i in xrange(len(result)):
		print "%s\t%s\t\t%s" % (kw, result[i], i)

def parse_result(html, filename):
	result = []

	try:
		soup = BeautifulSoup(html, "html.parser")
		#res_div_lst = soup.find_all("div", id="rg_s")

		#if len(res_div_lst) == 1:
		#	parse_img_search_div(res_div_lst[0], result)
		parse_img_search_div(soup, result)
		#else:
		#	print >>sys.stderr, "error rso_div: %s" % (filename)
	except Exception, e:
		print >>traceback.print_exc()
		print >>sys.stderr, "parse html error: %s" % (filename)

	return result

def parse_img_search_div_old(res_div, result):
	try:
		#print len(res_div.contents)
		#for ele_tag in res_div.find_all("div", class_="rg_di"):
		print >>sys.stderr, len(res_div.find_all("div"))
		for ele_tag in res_div.find_all("div", class_="rg_di"):
			#if not isinstance(ele_tag, element.Tag) or "rg_di" not in ele_tag["class"]:
			#	continue
			content_div = ele_tag.find_all("div", class_="rg_meta")[0]
			content = content_div.get_text()
			
			info = json.loads(content)
			fromurl = info['ru']
			objurl = info['ou']
			
			if fromurl and objurl:
				result.append("%s\1%s" % (fromurl, objurl))

	except Exception, e:
		print >>sys.stderr, "========"
		print >>sys.stderr, "parse img search div failed"
		print >>sys.stderr, traceback.print_exc()
		print >>sys.stderr, "========"

	return result

def parse_img_search_div(res_div, result):
	try:
		for ele_tag in res_div.find_all("div", class_="rg_meta"):
			#print >>sys.stderr, ele_tag
			#if not isinstance(ele_tag, element.Tag) or "rg_di" not in ele_tag["class"]:
			#	continue
			#content_div = ele_tag.find_all("div", class_="rg_meta")[0]
			content = ele_tag.get_text()
			
			info = json.loads(content)
			fromurl = info['ru']
			objurl = info['ou']
			
			if fromurl and objurl:
				result.append("%s" % (objurl))

			#break
	except Exception, e:
		print >>sys.stderr, "========"
		print >>sys.stderr, "parse img search div failed"
		print >>sys.stderr, traceback.print_exc()
		print >>sys.stderr, "========"

	return result

if __name__ == '__main__':
	if len(sys.argv) > 1:
		url = sys.argv[1]
		html_filename = sys.argv[2]
		parse_google_pages(url, html_filename)
	else:
		for line in sys.stdin:
			line = line.strip("\r\n")
			fs = line.split("\t")
			kw = fs[0].decode("utf-8").encode("gbk")
			index = fs[1]
			html_filename = "./special_google_pages/%s.html" % index
			parse_google_pages(kw, html_filename)
