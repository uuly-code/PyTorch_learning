true_box=[100,100,300,300]
pred_box=[200,200,400,400]

x_left=max(true_box[0],pred_box[0])
y_top=max(true_box[1],pred_box[1])

x_right=min(true_box[2],pred_box[2])
y_bottom=min(true_box[3],pred_box[3])

#计算重叠宽高
intersection_width=max(0,x_right-x_left)
intersection_height=max(0,y_bottom-y_top)

#计算重叠面积
intersection_area=(intersection_width*intersection_height)

#两个框各自的面积
true_area=((true_box[2]-true_box[0])*(true_box[3]-true_box[1]))
pred_area=(pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1])
union_area=(true_area+pred_area-intersection_area)

iou=intersection_area/union_area
print("IoU:",iou)