txt_name = 'val_chi.txt'
f = open(txt_name, "a+")
for i in range(0, 2914):
    f.write(str(i).zfill(4)+'\n')
f.close()

