#!/bin/sh
base_url="http://www.google.com.hk/search?biw=758&bih=643&site=imghp&tbm=isch&sa=1&"

function random()
{
	min=$1
	max=$2
	((margin=max-min))
	num=$(date +%s+%N)
	((ret_num=num%margin+min))
	return $ret_num
}

function wget_google_page()
{
	url=$1
	local_dest_file=$2
	for i in `seq 0 2`
	do
		echo $url 1>&2
		wget -Y on -e "http_proxy=http://agent.baidu.com:8118/" --user-agent="Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.204 Safari/534.16" --tries=3 -O $local_dest_file $url
		if [ $? -eq 0 ]
		then
			break
		fi
	done
}

while read line
do
	query=`echo "$line" | awk -F'\t' '{print $1}'`
	index=`echo "$line" | awk -F'\t' '{print $2}'`
	query=`echo "$query" | awk -F' ' '{a="";for(i=1;i<=NF;i++){if($i != " ")a=a"+"$i;}print substr(a, 2)}'`
	
	url="${base_url}q=${query}&op=${query}"
	local_filename="google_image_search_pages/${index}.html"
	wget_google_page $url $local_filename
	random "9" "20"
	sleep $?
done < $1

#wget -Y on -e "http_proxy=http://agent.baidu.com:8118/" --user-agent="Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.204 Safari/534.16" --tries=3 -O test_1.html "http://www.google.com.hk/searchbyimage?start=10&image_url=http%3A%2F%2Fatt.bbs.duowan.com%2Fforum%2F201606%2F12%2F165342nc3u86b7n636u1n6.jpg"
#wget -Y on -e "http_proxy=http://agent.baidu.com:8118/" --user-agent="Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.204 Safari/534.16" --tries=3 -O test_search_page.html $base_url
