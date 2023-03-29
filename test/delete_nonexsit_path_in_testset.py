import os.path

ori_test = '/raid/wangzihan/5th_ABAW/test/CVPR_5th_ABAW_AU_test_set_sample.txt'
out_test = '/raid/wangzihan/5th_ABAW/test/clear_test_set_path2.txt'



root  ='/raid/wangzihan/5th_ABAW/test/cropped_aligned/'
path = open(ori_test).readlines()

path = path[1:]

new_path = []
for i in range(len(path)):
    clear_path = path[i].strip('\n').strip(',')
    img_path = os.path.join(root, clear_path)
    if os.path.exists(img_path):
        new_path.append(clear_path)
        print(clear_path)

with open(out_test,'w') as f:
    for i in new_path:
        f.writelines(i)
        f.write('\n')