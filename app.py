import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import pickle
from datetime import timedelta , date

app=Flask(__name__)
model=pickle.load(open('model2.pkl','rb'))
temp=pickle.load(open("temperature.pkl","rb"))
humid=pickle.load(open("humidity.pkl","rb"))
windspd=pickle.load(open("windspeed.pkl","rb"))


@app.route('/')
def home():
    return render_template('index.html')  #html page here


@app.route('/predict',methods=['POST']) 
def predict():
    def convert(x):
        y=round(float(x),6)
        return y
    fcttemp=temp.predict(n_periods=5)
    fcttemp=pd.DataFrame(fcttemp,columns=['Prediction'])
    fcthumid=humid.predict(n_periods=5)
    fcthumid=pd.DataFrame(fcthumid,columns=['Prediction'])
    fctwindspd=windspd.predict(n_periods=5)
    fctwindspd=pd.DataFrame(fctwindspd,columns=['Prediction'])

    fcttemp.reset_index(drop=True,inplace=True)
    fcthumid.reset_index(drop=True,inplace=True)
    fctwindspd.reset_index(drop=True,inplace=True)
     
    day5=pd.DataFrame(columns=['Temperature','Humidity','Wind_Speed'])
    for i in range(0,5):
        row=[[convert(fcttemp.iloc[i].values),convert(fcthumid.iloc[i].values),convert(fctwindspd.iloc[i].values)]]
        day5=pd.concat([day5,pd.DataFrame(row,columns=['Temperature','Humidity','Wind_Speed'])],ignore_index=True)
    predict_climate = model.predict(day5)
    forecast=[]
    for i in range (0,5):
        end_date = date.today() + timedelta(days = i+1)
        end_date=end_date.strftime("%m/%d/%Y")
        c=day5.iloc[i].values
        forecast.append([end_date,{"Temperature":c[0] ,"Humidity":c[1],"WindSpeed":c[2]},model.predict(day5)[i]])

   


    int_features=[[int(x) for x in request.form.values()]]
    print(int_features)
    final_features=np.array(int_features)  
    prediction=model.predict(final_features)
    prediction=prediction.tolist()
    forecast=str(forecast)
    suggest = ""    
    if prediction == "Good":
        suggest = """If the climate remains stable and favorable, there are several positive impacts on the environment and society. 
• Agriculture could thrive with predictable growing seasons, leading to increased food production. 
• Ecosystems would be more resilient, supporting biodiversity and maintaining vital ecosystem services.
• Extreme weather events would be less frequent, reducing the risk of disasters.
• Sea levels and temperature fluctuations would remain relatively stable, preserving coastal communities and habitats.
To prevent negative impacts on the climate, it's crucial to continue efforts to reduce greenhouse gas emissions. 
• Transitioning to renewable energy sources, promoting sustainable land use, and enhancing energy efficiency are key strategies. 
• Conservation of forests and oceans also plays a vital role in maintaining a balanced climate."""
    elif prediction == "Moderate":
        suggest = """If the climate remains moderate, with balanced temperature and precipitation patterns, there could still be significant impacts on various aspects of life.
• Agricultural yields might experience fluctuations due to variations in growing conditions. 
• Ecosystems could face challenges in adapting to changing climate patterns, impacting biodiversity and ecosystem services. 

Preventing negative impacts in a moderate climate scenario involves many of the same strategies as in stable climates.
• Reducing emissions,
• Transitioning to renewable energy, 
• Practicing sustainable agriculture, and conserving natural resources."""
    else:       
        suggest= """A deteriorating climate can lead to a range of severe impacts, including more frequent and intense extreme weather events (such as hurricanes, droughts, and heatwaves), 
•	Rising sea levels that threaten coastal communities,
•	Disruptions to ecosystems and biodiversity,
•	Food and water scarcity, and negative health effects due to heat stress and air pollution. 
To prevent further climate degradation, actions must include 
•	Transitioning to renewable energy sources,
•	Adopting sustainable land use and forestry practices, 
•	Enhancing energy efficiency, promoting public transportation,
•	Implementing stricter emission regulations for industries, and fostering international cooperation to reduce greenhouse gas emissions and mitigate climate change effects.
"""

    suggest=str(suggest)
    #data = {          
        #   "Climate"   : forecast,
       #     "Recomendation" : suggest,
      ##return jsonify(data)
    
    return render_template('index.html',prediction_text='Your area Climate Now is  {}'.format(prediction))



if __name__=="__main__":
    app.run(debug=True)
    