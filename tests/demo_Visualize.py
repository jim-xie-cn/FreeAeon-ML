import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json,os,sys
import webbrowser
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from FreeAeonML.FAVisualize import CFAVisualize,CFANetGraph

#生成热力图
def test_heatmap():
    data_list = []
    data_list.append({"类别1": "A1", "类别2": "B1", "Count": 100})
    data_list.append({"类别1": "A2", "类别2": "B2", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B1", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B2", "Count": 10})
    df_data = pd.DataFrame(data_list)
    pt = df_data.pivot_table(values='Count', index=["类别1"], columns=['类别2'], aggfunc=np.sum, fill_value=0)
    CFAVisualize.show_heatmap(pt)

#生成桑基图
def test_sankey():
    data_list = []
    data_list.append({"类别1": "A1", "类别2": "B1", "Count": 100})
    data_list.append({"类别1": "A2", "类别2": "B2", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B1", "Count": 50})
    data_list.append({"类别1": "A3", "类别2": "B2", "Count": 10})
    df_data = pd.DataFrame(data_list)
    Sankey = CFAVisualize.get_sankey(df_data)
    html_file = './my_chart.html'
    file_path = os.path.abspath("my_chart.html")
    Sankey.render(file_path)  # 生成 HTML 文件
    webbrowser.open(f"file://{file_path}")

#生成等高线
def test_contour():
    data_list = []
    for i in range(10):
        for j in range(10):
            tmp = {"x": i, "y": j, "Count": np.sin(i * j)}
            data_list.append(tmp)
    df_data = pd.DataFrame(data_list)
    CFAVisualize.show_contour(df_data)

#生成序列图
def test_sequence():
    data = []
    for i in range(5):
        tmp = {}
        tmp['label'] = "name-%d"%i
        tmp['value'] = i
        tmp['title'] = "title-%d"%i

        data.append(tmp)
    df_data = pd.DataFrame(data)
    CFAVisualize.show_sequence(df_data)
    
def main():

    #生成热力图
    test_heatmap()
    plt.show()

    #生成等高线
    test_contour()
    plt.show()
    
    #生成桑基图
    test_sankey()
    
    #生成序列图
    test_sequence()

if __name__ == "__main__":
    main()