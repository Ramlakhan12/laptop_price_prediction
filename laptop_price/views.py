from django.shortcuts import render
from django.http import HttpResponse
import pickle as pkl
import pandas as pd
import numpy as np

# Create your views here.
def index(request):
    Memory_storage = [128, 500, 256,  64,  32, 512,   1,   2,  16, 180, 508,   8]
    Memory_storage.sort()
    ram = [16,  4,  8,  6,  2, 32, 12, 24, 64]
    ram.sort()

    ghz = [2.8 , 2.  , 2.5 , 1.1 , 1.6 , 2.7 , 2.9 , 2.4 , 1.44, 1.8 , 0.9 ,
       2.3 , 2.6 , 2.2 , 1.3 , 1.9 , 3.6 , 1.92, 3.  , 1.2 , 1.5 , 3.1 ,
       2.1 , 3.2 , 1.  ]
    ghz.sort()
    Storage_type = ['HDDSDD', 'HDD', 'SSD', 'Flash Storage', 'Hybrid']
    scaler = pkl.load(open(r'C:\Users\ramla\Desktop\django_projects\laptop_price_prediction\laptop_price\scaler.pkl','rb'))
    pipe = pkl.load(open(r'C:\Users\ramla\Desktop\django_projects\laptop_price_prediction\laptop_price\Pipe1.pkl','rb'))
    prediction_price = 0

    if request.method == 'POST':
        company = request.POST['Company']
        Memory1 = request.POST['Memory']
        Storage = request.POST['Storage']
        Brand = request.POST['Brand']
        core = request.POST['core_type']
        Inches = request.POST['Inches']
        Frequency = request.POST['Frequency']
        Storage_typ = request.POST['Storage_type']
        Weight = request.POST['Weight']
        scaled_data = scaler.transform([[Memory1,Storage,Weight]])
        
        print(company,Inches,Memory1,Storage,Storage_typ,Brand,core,Frequency,scaled_data[0][0],scaled_data[0][1],scaled_data[0][2])
        # input_data = np.array([company,Inches,Memory,Storage,Storage_typ,Brand,core,Frequency,scaled_data[0][0],scaled_data[0][1],scaled_data[0][2]],dtype=object).reshape(1,11)
        input_dict = {
            'Company': [company],
            'Inches': [Inches],
            'm_ram': [Memory1],
            'memo': [Storage],
            'Storage_Type': [Storage_typ],
            'Brand': [Brand],
            'Core Type': [core],
            'Frequency (GHz)': [Frequency],
            'n_ram': [scaled_data[0][0]],
            'n_memo': [scaled_data[0][1]],
            'wht': [scaled_data[0][2]]
        }
        input_df = pd.DataFrame(input_dict)
        prediction_price = np.expm1(pipe.predict(input_df))
    return render(request,'laptop_price/home.html',{'storage_list':Memory_storage,'memory_list':ram,'Heartz':ghz,'storage_type':Storage_type,'prediction':prediction_price})
