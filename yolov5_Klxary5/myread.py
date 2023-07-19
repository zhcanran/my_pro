import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load(r"D:\desktop\视觉算法\项目\危险品检测\yolov5_Klxary5\runs\train\exp\weights\best.onnx")
# model = onnx.load("yolov5s.onnx")

# for item in model.graph.initializer:
#   '''查看节点'''
    # if item.name == "model.0.conv.weight":
    #     print(item.name)
    #     dims = item.dims
    #     print('shape:',dims)
    #     data = np.frombuffer(item.raw_data, dtype=np.float32).reshape(dims)
    #     print(data.shape)

# for item in model.graph.node:
#     if item.op_type == 'Constant':
#         if '346' in item.output:
#             t = item.attribute[0].t
#             print(t.dims)
#             data = np.frombuffer(t.raw_data, dtype=np.float32).reshape(*t.dims)
#             print(data.shape)

# for item in model.graph.node:
#    '''修改节点值'''
#     if item.op_type == 'Constant':
#         if '352' in item.output:
#             t = item.attribute[0].t
#             data = np.frombuffer(t.raw_data, dtype=np.float32)
#             print(data)
#             t.raw_data = np.array([3.0],dtype=np.float32).tobytes()


# for item in model.graph.node:
#     if item.name == 'Reshape_216':
#         '''替换整个节点'''
#         print(item)
#         newitem = helper.make_node("Reshape",['356','452'],['363'],'Reshepe_555xxx')
#         item.CopyFrom(newitem)
#         print(item)


# find_node_with_input = lambda name: [item for item in model.graph.node if name in item.input][0]
# find_node_with_output = lambda name: [item for item in model.graph.node if name in item.output][0]
# remove_node =[]
# for item in model.graph.node:
#     if item.name == 'Transpose_200':
#         '''删除节点'''
#         #上一个节点的输出是当前节点的输入
#         prev = find_node_with_output(item.input[0])
#         # 下一个节点的输入是当前节点的输出
#         next = find_node_with_input(item.output[0])
#         next.input[0] = prev.output[0] #类似于链表操作
#         remove_node.apprend(item)
# for item in remove_node[::-1]:
#     model.graph.node.remove(item)


# 更改模型为静态或者动态
new_input = helper.make_tensor_value_info('images', 1, ['batch', 3, '640', '640'])
model.graph.input[0].CopyFrom(new_input)

new_output = helper.make_tensor_value_info('output0', 1, ['batch', '25200', '10'])
model.graph.output[0].CopyFrom(new_output)

onnx.save(model, 'best.onnx')
