from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

image_path= "detection_dataset/images/defect_01.png"
label_path= "detection_dataset/labels/defect_01.txt"

image=Image.open(image_path).convert("RGB")

image_width,image_height=image.size

print("图片宽度：",image_width)
print("图片高度：",image_height)

class_names={0:"scratch",1:"crack"}

with open(label_path,"r") as file:
    lines=file.readlines()

print("标签行数：",len(lines))

for line in lines:

    data=lines.split()
    class_id=int(data[0])

    #归一化的值
    x_center=float(data[1])
    y_center=float(data[2])
    box_width=float(data[3])
    box_height=float(data[4])

    #还原像素
    x_center=x_center*image_width
    y_center=y_center*image_height
    box_width=box_width*image_width
    box_height=box_height*image_height

    #转换格式
    x_min=x_center-box_width/2
    y_min=y_center-box_height/2

    x_max=x_center+box_width/2
    y_max=y_center+box_height/2

    #画出框

    draw=ImageDraw.Draw(image)
    draw.rectangle([x_min,y_min,x_max,y_max],outline="red",width=4)  #红色矩形框
    draw.text((x_min,y_min-15),"scratch",fill="red")


    plt.imshow(image)
    plt.axis("off")
    plt.show()
