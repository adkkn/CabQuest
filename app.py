from flask import Flask, render_template, request, jsonify
import folium
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.colors as mcolors
# import xgboost as xgb

app = Flask(__name__)

# Load the saved model
model = joblib.load('taxi_model.pkl')

totaltaxis = 12869

regions_data = [
    {"name": "Dubai Marina", "lat": 25.08078, "long": 55.14013, "radius": 1.7}, 
    {"name": "Jumeirah", "lat": 25.20295, "long": 55.24170, "radius": 2},  
    {"name": "Downtown Dubai / Business Bay", "lat": 25.18685, "long": 55.27390, "radius": 1.7}, 
    {"name": "Deira", "lat": 25.27958, "long": 55.33041, "radius": 2.8}, 
    {"name": "Palm Jumeirah", "lat": 25.11913, "long": 55.13157, "radius": 2.5},  
    {"name": "Jebel Ali", "lat": 25.01350, "long": 55.09690, "radius": 6.1},  
    # {"name": "Dubai Silicon Oasis", "lat": 25.12422, "long": 55.38496, "radius": 1.7},  
    {"name": "Mirdif", "lat": 25.21952, "long": 55.42481, "radius": 1.7},  
    {"name": "Al Quoz", "lat": 25.16135, "long": 55.25083, "radius": 1.9}, 
    {"name": "Dubai South", "lat": 24.885799, "long": 55.156770, "radius": 5.5},
    {"name": "Dubai Parks", "lat": 24.914563, "long": 55.011472, "radius": 1.9},
    {"name": "Al Barsha", "lat": 25.10164, "long": 55.20271, "radius": 2.2},
    {"name": "Internet City", "lat": 25.09878, "long": 55.16413, "radius": 1.45},
    {"name": "DIFC", "lat": 25.211724, "long": 55.274838, "radius": 1},
    {"name": "WTC", "lat": 25.22992, "long": 55.28963, "radius": 1.4},
    {"name": "Bur Dubai", "lat": 25.25464, "long": 55.29643, "radius": 1.4},
    {"name": "Airport / Garhoud / Festival City / Creek", "lat": 25.224273, "long": 55.351796, "radius": 3.7},
    {"name": "Al Qusais / Al Nahda / Muhaisnah", "lat": 25.281958, "long": 55.397610, "radius": 3.65},
    {"name": "Dubailand", "lat": 25.072645, "long": 55.306568, "radius": 6},
    {"name": "International City / Al Warqa", "lat": 25.173760, "long": 55.413906, "radius": 3.5},
]

taxi_ranks = [
    {"name": "RTA Taxi Rank - Al Mamzar, Century Mall", "lat": 25.289638, "long": 55.346959},
    {"name": "RTA Taxi Rank Hor Al Anz East, Abu Hail Center", "lat": 25.278079, "long": 55.346062},
    {"name": "RTA Taxi Rank Al Nahda -1, Al Mulla Plaza", "lat": 25.281348, "long": 55.357134},
    {"name": "RTA Taxi Rank Al Qusais -1, Al Bustan Center", "lat": 25.274128, "long": 55.367975},
    {"name": "RTA Taxi Rank Al Tawar -1, Union Coop Society", "lat": 25.2716, "long": 55.371133},
    {"name": "RTA Taxi Rank Mirdif, West Zone", "lat": 25.231387, "long": 55.418266},
    {"name": "RTA Taxi Rank Al Warqa -1, Al Mass Supermarket", "lat": 25.193895, "long": 55.406168},
    {"name": "RTA Taxi Rank Al Garhoud, Emirates Co op Socity", "lat": 25.243703, "long": 55.346579},
    {"name": "RTA Taxi Rank Al Muraqqbat, Al Ghurair City A", "lat": 25.266924, "long": 55.317057},
    {"name": "RTA Taxi Rank Al Muraqqbat, Al Ghurair City B", "lat": 25.268534, "long": 55.317864},
    {"name": "RTA Taxi Rank Riggat Al Buteen, Hilton Dubai Creek", "lat": 25.259693, "long": 55.318158},
    {"name": "RTA Taxi Rank Umm Hurair - 2, Raffles Hotel", "lat": 25.227824, "long": 55.321251},
    {"name": "RTA Taxi Rank Jumeirah - 3, Union Co-op Society", "lat": 25.187211, "long": 55.238977},
    {"name": "RTA Taxi Rank Business Bay, Downtown Burj Dubai - Building 3", "lat": 25.189962, "long": 55.280704},
    {"name": "RTA Taxi Rank Emirates Hills Second (Emaar Buildings) Emaar Business Park Buildings", "lat": 25.095254, "long": 55.166918},
    {"name": "RTA Taxi Rank Arabian Ranches Community Center", "lat": 25.05711, "long": 55.271085},
    {"name": "RTA Taxi Rank Al Barsha Mall ", "lat": 25.098519, "long": 55.205362},
    {"name": "RTA Taxi Rank Al Reef Mall ", "lat": 25.270147, "long": 55.322401},
    {"name": "RTA Taxi Rank Madina Mall ", "lat": 25.281958, "long": 55.39761},
    {"name": "RTA Taxi Rank Opp. Amwaj Rotana ", "lat": 25.073294, "long": 55.130559},
    {"name": "RTA Taxi Rank Sofitel", "lat": 25.075571, "long": 55.131841},
    {"name": "RTA Taxi Rank Opp. Le Royal Meridien ", "lat": 25.090815, "long": 55.147777},
    {"name": "RTA Taxi Rank Opp. Al Habtoor Grand Hotel", "lat": 25.085849, "long": 55.14174},
    {"name": "RTA Taxi Rank Opp. Grand Residence A", "lat": 25.083523, "long": 55.140719},
    {"name": "RTA Taxi Rank Opp. Grand Residence B", "lat": 25.083249, "long": 55.140057},
    {"name": "RTA Taxi Rank Opp. Sukoon Tower", "lat": 25.078832, "long": 55.144072},
    {"name": "RTA Taxi Rank Opp. Tram Number 6 ", "lat": 25.086546, "long": 55.149696},
    {"name": "RTA Taxi Rank Qusais, Lulu Hypermarket", "lat": 25.279075, "long": 55.361808},
    {"name": "RTA Taxi Rank Mall Of Emirates", "lat": 25.118107, "long": 55.200608},
    {"name": "RTA Taxi Rank Burjman Centre A", "lat": 25.252189, "long": 55.302387},
    {"name": "RTA Taxi Rank City Centre Deira", "lat": 25.250858, "long": 55.332997},
    {"name": "RTA Taxi Rank Festival City Mall", "lat": 25.224273, "long": 55.351796},
    {"name": "RTA Taxi Rank Dubai Marina Mall", "lat": 25.07643, "long": 55.140504},
    {"name": "RTA Taxi Rank Dubai Mall", "lat": 25.197642, "long": 55.27871},
    {"name": "RTA Taxi Rank Dragon Mart", "lat": 25.17376, "long": 55.413906},
    {"name": "RTA Taxi Rank Mercato Shopping Centre", "lat": 25.217066, "long": 55.252515},
    {"name": "RTA Taxi Rank City Centre Mirdif", "lat": 25.215272, "long": 55.409446},
    {"name": "RTA Taxi Rank Bin Sougat Shopping Centre", "lat": 25.232951, "long": 55.385168},
    {"name": "RTA Taxi Rank Ibn Batutta Mall Metro Station", "lat": 25.045643, "long": 55.117669},
    {"name": "RTA Taxi Rank Wafi Mall", "lat": 25.229532, "long": 55.319248},
    {"name": "RTA Taxi Rank Dubai Outlet Village", "lat": 24.914563, "long": 55.011472},
    {"name": "RTA Taxi Rank Al Rigga Metro Station A", "lat": 25.263424, "long": 55.323447},
    {"name": "RTA Taxi Rank Union Metro Station", "lat": 25.266278, "long": 55.315183},
    {"name": "RTA Taxi Rank Burjuman Metro Station D", "lat": 25.255438, "long": 55.304969},
    {"name": "RTA Taxi Rank ADCB Metro Station Seaside", "lat": 25.245147, "long": 55.298246},
    {"name": "RTA Taxi Rank ADCB Metro Station Landside", "lat": 25.245274, "long": 55.298922},
    {"name": "RTA Taxi Rank Max Metro Station Seaside", "lat": 25.233136, "long": 55.290629},
    {"name": "RTA Taxi Rank Max Metro Station Landside B", "lat": 25.231416, "long": 55.292624},
    {"name": "RTA Taxi Rank World Trade Centre Metro Station Seaside", "lat": 25.224631, "long": 55.283507},
    {"name": "RTA Taxi Rank World Trade Centre Metro Station Landside", "lat": 25.225808, "long": 55.285875},
    {"name": "RTA Taxi Rank Emirates Towers Metro Station Landside", "lat": 25.216855, "long": 55.280354},
    {"name": "RTA Taxi Rank Financial Centre Metro Station Seaside", "lat": 25.211724, "long": 55.274838},
    {"name": "RTA Taxi Rank Financial Centre Metro Station Landside", "lat": 25.211297, "long": 55.276043},
    {"name": "RTA Taxi Rank Burj Khalifa/ Dubai Mall Metro Station Landside A", "lat": 25.200089, "long": 55.268402},
    {"name": "RTA Taxi Rank Burj Khalifa/ Dubai Mall Metro Station Landside B", "lat": 25.200639, "long": 55.269566},
    {"name": "RTA Taxi Rank Business Bay Metro Station Seaside", "lat": 25.192325, "long": 55.259923},
    {"name": "RTA Taxi Rank Business Bay Metro Station Landside", "lat": 25.190709, "long": 55.260044},
    {"name": "RTA Taxi Rank Mall Of The Emirates Metro Station Landside", "lat": 25.120161, "long": 55.201598},
    {"name": "RTA Taxi Rank Mashreq Metro Station Seaside", "lat": 25.115386, "long": 55.190077},
    {"name": "RTA Taxi Rank Mashreq Metro Station Landside", "lat": 25.11459, "long": 55.191135},
    {"name": "RTA Taxi Rank Dubai Internet City Metro Station Landside", "lat": 25.099459, "long": 55.172341},
    {"name": "RTA Taxi Rank Dubai Internet City Metro Station Seaside", "lat": 25.101936, "long": 55.171947},
    {"name": "RTA Taxi Rank Al Khail Metro Station Seaside", "lat": 25.089324, "long": 55.156815},
    {"name": "RTA Taxi Rank Al Khail Metro Station Landside", "lat": 25.089288, "long": 55.158908},
    {"name": "RTA Taxi Rank Sobha Realty Metro Station Seaside", "lat": 25.081455, "long": 55.14712},
    {"name": "RTA Taxi Rank DMCC Metro Station Seaside", "lat": 25.071525, "long": 55.137381},
    {"name": "RTA Taxi Rank Energy Metro Station Seaside", "lat": 25.026041, "long": 55.09971},
    {"name": "RTA Taxi Rank Energy Metro Station Landside", "lat": 25.025308, "long": 55.101136},
    {"name": "RTA Taxi Rank Danube Metro Station Landside", "lat": 25.000319, "long": 55.09595},
    {"name": "RTA Taxi Rank Dubai Airport Free Zone Metro Station 1", "lat": 25.268897, "long": 55.375557},
    {"name": "RTA Taxi Rank Al Qusais Metro Station 1", "lat": 25.262687, "long": 55.386166},
    {"name": "RTA Taxi Rank Al Nahda Metro Station 1", "lat": 25.273472, "long": 55.368161},
    {"name": "RTA Taxi Rank Stadium Metro Station 1", "lat": 25.276397, "long": 55.362748},
    {"name": "RTA Taxi Rank Al Qiyadah Metro Station Seaside", "lat": 25.279183, "long": 55.352021},
    {"name": "RTA Taxi Rank Abu Hail Metro Station Seaside A", "lat": 25.276212, "long": 55.34732},
    {"name": "RTA Taxi Rank Abu Hail Metro Station B", "lat": 25.27624, "long": 55.346146},
    {"name": "RTA Taxi Rank Abu Hail Metro Station C", "lat": 25.275307, "long": 55.347157},
    {"name": "RTA Taxi Rank Sharaf DG Metro Station B 2", "lat": 25.257434, "long": 55.296989},
    {"name": "RTA Taxi Rank Creek Metro Station", "lat": 25.219494, "long": 55.338303},
    {"name": "RTA Taxi Rank UAE Exchange Metro Station, Seaside  (Inside Jebel Ali Port)", "lat": 24.97767, "long": 55.088664},
    {"name": "RTA Taxi Rank UAE Exchange, Landside", "lat": 24.976919, "long": 55.09116},
    {"name": "RTA Taxi Rank The Palm Jumeirah", "lat": 25.116871, "long": 55.136336},
    {"name": "RTA Taxi Rank King Salman Bin Abdulaziz Al Soud St.", "lat": 25.094947, "long": 55.154394},
    {"name": "RTA Taxi Rank Al Thaniya Street Barsha Heights", "lat": 25.095731, "long": 55.177456},
    {"name": "RTA Taxi Rank Sama Building , Al Barsha", "lat": 25.109234, "long": 55.203023},
    {"name": "RTA Taxi Rank Umm Suqeim Street", "lat": 25.111982, "long": 55.218784},
    {"name": "RTA Taxi Rank Ghoroob, Mirdif", "lat": 25.211504, "long": 55.417871},
    {"name": "RTA Taxi Rank Mankhool Road Near Standard Chartered Tower", "lat": 25.25246, "long": 55.291681},
    {"name": "RTA Taxi Rank Garhoud Near Millennium Airport Hotel", "lat": 25.250201, "long": 55.344459},
    {"name": "RTA Taxi Rank Khaleej Road, Waterfront Market", "lat": 25.291471, "long": 55.323995},
    {"name": "RTA Taxi Rank Behind Sahara Center, Al Nahda 1", "lat": 25.296851, "long": 55.370199},
    {"name": "RTA Taxi Rank Near Dubai Carmel School, Al Amman Street", "lat": 25.298405, "long": 55.380486},
    {"name": "RTA Taxi Rank Mirdif, Uptown Shopping Centre", "lat": 25.224303, "long": 55.424004},
    {"name": "RTA Taxi Rank Vida Downtown Dubai", "lat": 25.18993, "long": 55.27416},
    {"name": "RTA Taxi Rank Shangri-La Hotel", "lat": 25.208492, "long": 55.272459},
    {"name": "RTA Taxi Rank Gloria Hotel ", "lat": 25.103122, "long": 55.17139},
    {"name": "RTA Taxi Rank Grand Millennium Hotel Dubai", "lat": 25.101627, "long": 55.177249},
    {"name": "RTA Taxi Rank Oasis Center", "lat": 25.170001, "long": 55.24116},
    {"name": "RTA Taxi Rank Al Maktoum International Airport", "lat": 24.885799, "long": 55.15677},
    {"name": "RTA Taxi Rank Bollywood Parks Dubai Park & Resorts", "lat": 24.915753, "long": 55.007915},
    {"name": "RTA Taxi Rank Motiongate Dubai- Dubai Parks And Resorts ", "lat": 24.921158, "long": 55.008388},
    {"name": "RTA Taxi Rank Global Village", "lat": 25.072645, "long": 55.306568},
    {"name": "RTA Taxi Rank Emirates HQ", "lat": 25.242019, "long": 55.366098},
    {"name": "RTA Taxi Rank Four Seasons Resort Dubai At Jumeirah Beav", "lat": 25.202235, "long": 55.240601},
    {"name": "RTA Taxi Rank Al Ghubaiba Bus Station Backside", "lat": 25.261755, "long": 55.288714},
    {"name": "RTA Taxi Rank Discovery Garden 1", "lat": 25.044048, "long": 55.134959},
    {"name": "RTA Taxi Rank Discovery Garden 2", "lat": 25.044164, "long": 55.133608},
    {"name": "RTA Taxi Rank Discovery Garden 3", "lat": 25.035241, "long": 55.145644},
    {"name": "RTA Taxi Rank Discovery Garden 4", "lat": 25.034153, "long": 55.146393},
    {"name": "RTA Taxi Rank Discovery Garden 5", "lat": 25.030705, "long": 55.152477},
    {"name": "RTA Taxi Rank Discovery Garden 6", "lat": 25.030818, "long": 55.151135},
    {"name": "RTA Taxi Rank Souq Naif", "lat": 25.271211, "long": 55.305302},
    {"name": "RTA Taxi Rank Dusit Thani Hotel", "lat": 25.206799, "long": 55.272998},
    {"name": "RTA Taxi Rank Carlton Downtown Hotel", "lat": 25.209235, "long": 55.274458},
    {"name": "RTA Taxi Rank Grand Stay Hotel", "lat": 25.210024, "long": 55.274986},
    {"name": "RTA Taxi Rank Financial Centre Metro Station B", "lat": 25.210274, "long": 55.276242},
    {"name": "RTA Taxi Rank Rose Rayhaan By Rotana Hotel", "lat": 25.21207, "long": 55.276378},
    {"name": "RTA Taxi Rank Gevora Hotel", "lat": 25.212538, "long": 55.276713},
    {"name": "RTA Taxi Rank DIP Metro Station Gate2A", "lat": 25.005012, "long": 55.155437},
    {"name": "RTA Taxi Rank Burjuman Center B", "lat": 25.252164, "long": 55.302416},
    {"name": "RTA Taxi Rank IKEA - Festival Center A", "lat": 25.223393, "long": 55.35665},
    {"name": "RTA Taxi Rank IKEA - Festival Center B", "lat": 25.22314, "long": 55.356545},
    {"name": "RTA Taxi Rank Discovery Gardens", "lat": 25.04911355, "long": 55.12793376},
    {"name": "RTA Taxi Rank JLT Gold Tower", "lat": 25.06930493, "long": 55.14349716},
    {"name": "RTA Taxi Rank Near Al Zahra Hospital", "lat": 25.10686284, "long": 55.18202891},
    {"name": "RTA Taxi Rank Deyaar Development PJSC (Sales Center)", "lat": 25.1858492, "long": 55.26527169},
    {"name": "RTA Taxi Rank Burj Khalifa/Dubai Mall Metro Station D", "lat": 25.19961089, "long": 55.26961725},
    {"name": "RTA Taxi Rank Sheraton Grand Hotel", "lat": 25.22940049, "long": 55.2869376},
    {"name": "RTA Taxi Rank Canadian Specialist Hospital", "lat": 25.27793672, "long": 55.34886644},
    {"name": "RTA Taxi Rank Endocrinologist Dubai Prime Hospital", "lat": 25.24947921, "long": 55.3455928},
    {"name": "RTA Taxi Rank Mediclinic Welcare Hospital", "lat": 25.24727744, "long": 55.33958297},
    {"name": "RTA Taxi Rank Circle K Store Holiday Inn Hotel", "lat": 25.24222326, "long": 55.36081911},
    {"name": "RTA Taxi Rank Dubai Residential Oasis", "lat": 25.28286469, "long": 55.38868009},
    {"name": "RTA Taxi Rank Dubai Grand Hotel By Fortune", "lat": 25.27319408, "long": 55.38145335},
    {"name": "RTA Taxi Rank Lulu Village", "lat": 25.28142523, "long": 55.41129326},
    {"name": "RTA Taxi Rank Al Ghubaibah Taxi Station Backside", "lat": 25.261776, "long": 55.28874},
    {"name": "RTA Taxi Rank Internet City Al Jaddi Street", "lat": 25.101013, "long": 55.16942},
    {"name": "RTA Taxi Rank Ansar Gallery", "lat": 25.252236, "long": 55.309019},
    {"name": "RTA Taxi Rank Dubai Frame Zabeel Park", "lat": 25.234252, "long": 55.301575},
    {"name": "RTA Taxi Rank DIP Metro Station Gate 2B", "lat": 25.004211, "long": 55.155189},
    {"name": "RTA Taxi Rank DIP Metro Station Gate 2A", "lat": 25.006111, "long": 55.15568},
    {"name": "RTA Taxi Rank DIP Metro Station Gate 1", "lat": 25.0041, "long": 55.155572},
    {"name": "RTA Taxi Rank RAK Bank Near Burjuman Metro Station", "lat": 25.256451, "long": 55.305663},
    {"name": "RTA Taxi Rank Jebel Ali Metro Station Landside", "lat": 25.057292, "long": 55.127001},
    {"name": "RTA Taxi Rank Deira Palm, Night Market 1", "lat": 25.29166, "long": 55.311688},
    {"name": "RTA Taxi Rank Seniors Happiness Centre", "lat": 25.298022, "long": 55.345618},
    {"name": "RTA Taxi Rank Al Nahda Fitness Centre", "lat": 25.28258, "long": 55.372335},
    {"name": "RTA Taxi Rank Claridge Hotel Dubai ", "lat": 25.270999, "long": 55.316488},
    {"name": "RTA Taxi Rank Salah Al Din Metro Station 2", "lat": 25.270472, "long": 55.318509},
    {"name": "RTA Taxi Rank Abu Baker Al Siddique Metro Station 1", "lat": 25.2708, "long": 55.333412},
    {"name": "RTA Taxi Rank Duja Tower ", "lat": 25.229243, "long": 55.286041},
    {"name": "RTA Taxi Rank Dubai Court Personal Status Court", "lat": 25.23505, "long": 55.355183},
    {"name": "RTA Taxi Rank Commercial Tower", "lat": 25.221078, "long": 55.28105},
    {"name": "RTA Taxi Rank DoubleTree By Hilton Jumeirah", "lat": 25.071991, "long": 55.127904},
    {"name": "RTA Taxi Rank City Star Hotel", "lat": 25.109086, "long": 55.203145},
    {"name": "RTA Taxi Rank Emarat Petrol HQ", "lat": 25.191612, "long":55.258374},
]


# Function to generate 30-minute time intervals
def generate_time_intervals():
    return [(hour, minute) for hour in range(24) for minute in [0, 30]]

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_time = None

    if request.method == 'POST':
        selected_time = request.form['time']
        print(f"Selected time: {selected_time}")

    return render_template(
        'index.html', 
        time_intervals=generate_time_intervals(), 
        selected_time=selected_time
    )

@app.route('/dubai-map')
def dubai_map():
    # Center the map on Dubai
    dubai_coordinates = [25.276987, 55.296249]  # Latitude and Longitude of Dubai

    # Create a Folium map
    folium_map = folium.Map(location=dubai_coordinates, zoom_start=11, tiles="OpenStreetMap", scrollWheelZoom=False)

    # Add markers for each taxi rank
    for rank in taxi_ranks:
        folium.Marker(
            location=[rank["lat"], rank["long"]],
            popup=rank["name"],
            icon=folium.Icon(color="blue", icon="info-sign")  # Simpler marker icon
        ).add_to(folium_map)

    return folium_map._repr_html_()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json().get('time', 0)
        print(f"Input time received from frontend: {input_data}")

        # Convert time to the expected format for the model
        time_in_minutes = int(input_data) * 60
        time_for_prediction = pd.DataFrame({'Hour': [time_in_minutes]})

        # Get prediction probabilities
        predicted_region_proba = model.predict_proba(time_for_prediction)[0]
        print("Prediction results: ", predicted_region_proba)

        # Find minimum and maximum probabilities to normalize the color range
        min_proba = np.min(predicted_region_proba)
        max_proba = np.max(predicted_region_proba)

        # Normalize the probabilities
        normalized_proba = (predicted_region_proba - min_proba) / (max_proba - min_proba + 1e-6)

        # Ensure correct label mapping
        label_mapping = {
            0: "Airport / Garhoud / Festival City / Creek",
            1: "Al Barsha",
            2: "Al Quoz",
            3: "Al Qusais / Al Nahda / Muhaisnah",
            4: "Bur Dubai",
            5: "DIFC",
            6: "Deira",
            7: "Downtown Dubai / Business Bay",
            8: "Dubai Marina",
            9: "Dubai Parks",
            10: "Dubai Silicon Oasis",
            11: "Dubai South",
            12: "Dubailand",
            13: "International City / Al Warqa",
            14: "Internet City",
            15: "Jebel Ali",
            16: "Jumeirah",
            17: "Mirdif",
            18: "Palm Jumeirah",
            19: "WTC"
        }

        # Generate the map
        dubai_coordinates = [25.276987, 55.296249]  # Dubai coordinates
        folium_map = folium.Map(location=dubai_coordinates, zoom_start=11, tiles="OpenStreetMap", scrollWheelZoom=False)

        # Create color gradient based on probability (green = low, red = high)
        for i, region in enumerate(regions_data):
            probability = normalized_proba[i]
            # Generate color gradient
            color = mcolors.to_hex(mcolors.LinearSegmentedColormap.from_list("", ["green", "yellow", "red"])(probability))
            
            # Add circle with color based on probability
            folium.Circle(
                location=[region["lat"], region["long"]],
                radius=region["radius"] * 1000,  # Convert km to meters
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{region['name']}: {probability * 100:.2f}% demand"
            ).add_to(folium_map)

        # Return the map as HTML
        return folium_map._repr_html_()

    except Exception as e:
        print(f"Error while making predictions: {e}")
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
