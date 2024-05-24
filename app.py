import aiohttp
from flask import Flask, jsonify, request
import numpy as np
from keras.models import load_model
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

model = load_model('SleepApnea_SpO2.h5')
api_key = "BBUS-llkCCBCGG4YJGJjsl12mywraAcfQkV"
latest_timestamp = None 
spO2_values = [] 
terminate_flag = False  
communication_broken = False 
heartbeat_received = True  

async def fetch_latest_spO2_values_from_ubidots(api_key, latest_timestamp, spO2_values):
    live_count = 0
    url = 'https://industrial.api.ubidots.com/api/v1.6/devices/apneasense/spo2/values/?page_size=1'
    headers = {"X-Auth-Token": api_key}
    global terminate_flag, communication_broken

    async with aiohttp.ClientSession() as session:
        while not terminate_flag:  
            if communication_broken:
                spO2_values.clear() 
                return None
            try:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if "results" in data and data["results"]:
                        latest_entry = data["results"][0]
                        if latest_entry['timestamp'] != latest_timestamp:
                            spO2_values.append(latest_entry['value'])
                            live_count += 1
                            print("Element Added :: ", live_count)
                            if len(spO2_values) == 88:
                                temp = spO2_values.copy()
                                spO2_values.clear()
                                return temp
                            latest_timestamp = latest_entry['timestamp']
            except aiohttp.ClientError as e:
                print("Error fetching SpO2 values:", e)
                return None  
        terminate_flag = False    
        print('Loop Terminated')
        return False

@app.route('/spo2predict', methods=['GET'])
async def predict():
    spo2_values_fetched = await fetch_latest_spO2_values_from_ubidots(api_key, latest_timestamp, spO2_values)
    if spo2_values_fetched is None:
        return jsonify({'message': 'Communication with the server is currently broken. Please try again later.'}), 503
    elif not spo2_values_fetched:
        return jsonify({'message': 'Process Terminated.'}), 200 
       
    reshaped_values = np.array(spo2_values_fetched).reshape((1, 88))
    mean_test = np.mean(reshaped_values)
    std_test = np.std(reshaped_values)
    spo2_test_normalized = (reshaped_values - mean_test) / std_test
    spo2_test_normalized = np.expand_dims(spo2_test_normalized, axis=2)
    y_pred = model.predict(spo2_test_normalized)
    predict_test = np.argmax(y_pred, axis=1)
    predict_test = predict_test.reshape(predict_test.shape[0], 1)
    array_as_list = predict_test.tolist()
    return jsonify({'array': array_as_list}), 200

@app.route('/spo2terminate', methods=['POST'])
def terminate():
    global terminate_flag, spO2_values
    terminate_flag = True
    spO2_values.clear() 
    return jsonify({'message': 'Operation Successful.'}), 200

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    print('Heart Beat ...')
    global heartbeat_received, communication_broken
    heartbeat_received = True
    communication_broken = False  
    return jsonify({'message': 'Heartbeat received.'}), 200

def check_heartbeat():
    global heartbeat_received, communication_broken
    if not heartbeat_received:
        communication_broken = True
        print("Communication with Flutter app is broken.")
    heartbeat_received = False

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=check_heartbeat, trigger="interval", seconds=10)
    scheduler.start()
    
    app.run(host='0.0.0.0', port=5000)
