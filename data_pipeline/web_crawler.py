import os
import requests
import json
import time
import urllib.parse

# 查询的关键词
query_name = 'hanhong'
encoded_query_name = requests.utils.quote(query_name)

# 目标URL模板
url1 = 'https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1716975508081_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=MCwxLDMsMiw2LDQsNSw4LDcsOQ%3D%3D&ie=utf-8&sid=&word=%E5%BC%A0%E5%9B%BD%E7%AB%8B'

parsed_url = urllib.parse.urlparse(url1)
query_params = urllib.parse.parse_qs(parsed_url.query)
word_param = query_params.get('word', [''])[0]
encoded_query_name = urllib.parse.quote(word_param)

url_template = f'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=111&ipn=rj&ct=201326592&is=&fp=result&fr=&word={encoded_query_name}&cg=star&pn='


# 存储照片的文件夹路径
base_save_path = 'E:\CV_Data\Face-EEG\StarImg\chineseface'
save_path = os.path.join(base_save_path, query_name)

# 创建存储照片的文件夹
os.makedirs(save_path, exist_ok=True)


def fetch_image_urls(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return set()

    img_urls = set()
    try:
        json_data = response.json()
        for item in json_data.get('data', []):
            if 'thumbURL' in item:
                img_urls.add(item['thumbURL'])
    except json.JSONDecodeError:
        pass

    return img_urls


def download_images(img_urls, save_path, start_count, target_count=200):
    count = start_count
    img_urls = list(img_urls)
    while count < target_count and img_urls:
        img_url = img_urls.pop(0)
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            img_path = os.path.join(save_path, f'{query_name}_{count + 1}.jpg')
            with open(img_path, 'wb') as f:
                f.write(img_response.content)
            count += 1
        else:
            print(f"下载失败，照片URL: {img_url}")

    return count

# Headers for the request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}

# 确保下载到200张图片
total_downloaded = len(os.listdir(save_path))  # 检查已有图片数量
page_num = 0
downloaded_urls = set()

while total_downloaded < 200:
    url = url_template + str(page_num * 30)
    img_urls = fetch_image_urls(url, headers)
    print(f"找到的图片URL数量: {len(img_urls)}")

    # 过滤已经下载过的URL
    new_img_urls = img_urls - downloaded_urls
    downloaded_urls.update(new_img_urls)

    if not new_img_urls:
        print("没有更多图片可下载")
        break

    total_downloaded = download_images(new_img_urls, save_path, total_downloaded, 200)
    print(f"成功下载了{total_downloaded}张照片")

    if total_downloaded < 200:
        page_num += 1  # 增加页数
        time.sleep(1)  # 增加延时以防止被封禁

print(f"最终成功下载了{total_downloaded}张照片")
