
clear_img_path = open('/raid/wangzihan/5th_ABAW/test/clear_test_set_path.txt').readlines()

result_img_path = open('/raid/wangzihan/5th_ABAW/test/CVPR_5th_ABAW_AU_test_set_sample.txt').readlines()

head = result_img_path[0]
result_img_path = result_img_path[1:]

clear_pred = open('/raid/wangzihan/5th_ABAW/test/clear_result_4.txt').readlines()

result = []

with open('/raid/wangzihan/5th_ABAW/test/fin_result_4.txt','w') as f:
    f.writelines(head)
    i, j = 0, 0
    while j < len(result_img_path):
        if clear_img_path[i].strip() == result_img_path[j].strip().strip(','):
            f.writelines(clear_img_path[i].strip()+','+clear_pred[i])
            i+=1
            j+=1
        else:
            f.writelines(result_img_path[j].strip()+clear_pred[i])
            j+=1
