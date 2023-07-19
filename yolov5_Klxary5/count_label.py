import os
import collections

url=r"D:\desktop\项目\危险品检测\Slxary\labels\val"
file_dir = os.listdir(url)


count = collections.defaultdict(int)
print(len(file_dir))
count1=[]
for i in file_dir:
    with open(os.path.join(url, i),'r') as f:
        data = f.readlines()
    for label in data:
        count1.append(label[0])
        count[label[0]] += 1
    
print(collections.Counter(count1))
print(count)
    
# Counter({'0': 3675, '2': 3331, '1': 2055, '4': 2035, '3': 743})
# Counter({'0': 1695, '2': 1647, '4': 1044, '1': 1029, '3': 389})