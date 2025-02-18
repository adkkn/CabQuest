from flask import Flask, render_template, request, jsonify
import folium
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.colors as mcolors
import google.generativeai as genai

app = Flask(__name__)

# Load the saved model
model = joblib.load('taxi_model.pkl')

totaltaxis = 12869

regions_data = [
    {"name": "Airport / Garhoud / Festival City / Creek", "lat": 25.224273, "long": 55.351796, "radius": 3.7},
    {"name": "Al Barsha", "lat": 25.10164, "long": 55.20271, "radius": 2.2},
    {"name": "Al Quoz", "lat": 25.16135, "long": 55.25083, "radius": 1.9},
    {"name": "Al Qusais / Al Nahda / Muhaisnah", "lat": 25.281958, "long": 55.397610, "radius": 3.65},
    {"name": "Bur Dubai", "lat": 25.25464, "long": 55.29643, "radius": 1.4},
    {"name": "Deira", "lat": 25.27958, "long": 55.33041, "radius": 2.8},
    {"name": "DIFC", "lat": 25.211724, "long": 55.274838, "radius": 1},
    {"name": "Downtown Dubai / Business Bay", "lat": 25.18685, "long": 55.27390, "radius": 1.7},
    {"name": "Dubai Marina", "lat": 25.08078, "long": 55.14013, "radius": 1.7},
    {"name": "Dubai Parks", "lat": 24.914563, "long": 55.011472, "radius": 1.9},
    {"name": "Dubai South", "lat": 24.885799, "long": 55.156770, "radius": 5.5},
    {"name": "Dubailand", "lat": 25.072645, "long": 55.306568, "radius": 6},
    {"name": "International City / Al Warqa", "lat": 25.173760, "long": 55.413906, "radius": 3.5},
    {"name": "Internet City", "lat": 25.09878, "long": 55.16413, "radius": 1.45},
    {"name": "Jebel Ali", "lat": 25.01350, "long": 55.09690, "radius": 6.1},
    {"name": "Jumeirah", "lat": 25.20295, "long": 55.24170, "radius": 2},
    {"name": "Mirdif", "lat": 25.21952, "long": 55.42481, "radius": 1.7},
    {"name": "Palm Jumeirah", "lat": 25.11913, "long": 55.13157, "radius": 2.5},
    {"name": "WTC", "lat": 25.22992, "long": 55.28963, "radius": 1.4}
]

taxi_ranks = [{"region": "Airport / Garhoud / Festival City / Creek", "ratio": 18.637654753562245, "name": "RTA Taxi Rank Al Garhoud, Emirates Co op Socity", "lat": 25.243703, "long": 55.346579}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 65.53239897220178, "name": "RTA Taxi Rank Umm Hurair - 2, Raffles Hotel", "lat": 25.227824, "long": 55.321251}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 135.874515300163, "name": "RTA Taxi Rank City Centre Deira", "lat": 25.250858, "long": 55.332997}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 217.03849567857887, "name": "RTA Taxi Rank Festival City Mall", "lat": 25.224273, "long": 55.351796}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 15.63158140621275, "name": "RTA Taxi Rank Bin Sougat Shopping Centre", "lat": 25.232951, "long": 55.385168}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 246.49801448259694, "name": "RTA Taxi Rank Wafi Mall", "lat": 25.229532, "long": 55.319248}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 18.036440084091833, "name": "RTA Taxi Rank Creek Metro Station", "lat": 25.219494, "long": 55.338303}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 83.56883905629489, "name": "RTA Taxi Rank Garhoud Near Millennium Airport Hotel", "lat": 25.250201, "long": 55.344459}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 21.643728100910458, "name": "RTA Taxi Rank Emirates HQ", "lat": 25.242019, "long": 55.366098}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 204.4129876197143, "name": "RTA Taxi Rank IKEA - Festival Center A", "lat": 25.223393, "long": 55.35665}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 204.4129876197143, "name": "RTA Taxi Rank IKEA - Festival Center B", "lat": 25.22314, "long": 55.356545}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 37.87652417659362, "name": "RTA Taxi Rank Endocrinologist Dubai Prime Hospital", "lat": 25.24947921, "long": 55.3455928}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 33.66802149030587, "name": "RTA Taxi Rank Mediclinic Welcare Hospital", "lat": 25.24727744, "long": 55.33958297}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 46.29352954917041, "name": "RTA Taxi Rank Circle K Store Holiday Inn Hotel", "lat": 25.24222326, "long": 55.36081911}, {"region": "Airport / Garhoud / Festival City / Creek", "ratio": 9.619434711515042, "name": "RTA Taxi Rank Dubai Court Personal Status Court", "lat": 25.23505, "long": 55.355183}, {"region": "Al Barsha", "ratio": 195.4438453944859, "name": "RTA Taxi Rank Al Barsha Mall ", "lat": 25.098519, "long": 55.205362}, {"region": "Al Barsha", "ratio": 16.612726858531378, "name": "RTA Taxi Rank Mall Of Emirates", "lat": 25.118107, "long": 55.200608}, {"region": "Al Barsha", "ratio": 174.92224162806517, "name": "RTA Taxi Rank Mall Of The Emirates Metro Station Landside", "lat": 25.120161, "long": 55.201598}, {"region": "Al Barsha", "ratio": 35.17989217100679, "name": "RTA Taxi Rank Mashreq Metro Station Seaside", "lat": 25.115386, "long": 55.190077}, {"region": "Al Barsha", "ratio": 439.7486521375958, "name": "RTA Taxi Rank Mashreq Metro Station Landside", "lat": 25.11459, "long": 55.191135}, {"region": "Al Barsha", "ratio": 193.48940694054184, "name": "RTA Taxi Rank Sama Building , Al Barsha", "lat": 25.109234, "long": 55.203023}, {"region": "Al Barsha", "ratio": 19.544384539448075, "name": "RTA Taxi Rank Umm Suqeim Street", "lat": 25.111982, "long": 55.218784}, {"region": "Al Barsha", "ratio": 30.293796036144773, "name": "RTA Taxi Rank Near Al Zahra Hospital", "lat": 25.10686284, "long": 55.18202891}, {"region": "Al Barsha", "ratio": 214.98822993393526, "name": "RTA Taxi Rank City Star Hotel", "lat": 25.109086, "long": 55.203145}, {"region": "Al Quoz", "ratio": 988.8364434687148, "name": "RTA Taxi Rank Oasis Center", "lat": 25.170001, "long": 55.24116}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 68.82869804400966, "name": "RTA Taxi Rank Al Qusais -1, Al Bustan Center", "lat": 25.274128, "long": 55.367975}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 12.782472493886477, "name": "RTA Taxi Rank Al Tawar -1, Union Coop Society", "lat": 25.2716, "long": 55.371133}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 14.749006723716352, "name": "RTA Taxi Rank Madina Mall ", "lat": 25.281958, "long": 55.39761}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 277.2813264058672, "name": "RTA Taxi Rank Qusais, Lulu Hypermarket", "lat": 25.279075, "long": 55.361808}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 133.7243276283608, "name": "RTA Taxi Rank Dubai Airport Free Zone Metro Station 1", "lat": 25.268897, "long": 55.375557}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 259.5825183374073, "name": "RTA Taxi Rank Al Qusais Metro Station 1", "lat": 25.262687, "long": 55.386166}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 79.64463630806753, "name": "RTA Taxi Rank Al Nahda Metro Station 1", "lat": 25.273472, "long": 55.368161}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 41.29721882640553, "name": "RTA Taxi Rank Stadium Metro Station 1", "lat": 25.276397, "long": 55.362748}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 897.7228759168702, "name": "RTA Taxi Rank Behind Sahara Center, Al Nahda 1", "lat": 25.296851, "long": 55.370199}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 547.6797830073349, "name": "RTA Taxi Rank Near Dubai Carmel School, Al Amman Street", "lat": 25.298405, "long": 55.380486}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 67.8454309290947, "name": "RTA Taxi Rank Dubai Residential Oasis", "lat": 25.28286469, "long": 55.38868009}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 140.6071974327628, "name": "RTA Taxi Rank Dubai Grand Hotel By Fortune", "lat": 25.27319408, "long": 55.38145335}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 21.63187652811706, "name": "RTA Taxi Rank Lulu Village", "lat": 25.28142523, "long": 55.41129326}, {"region": "Al Qusais / Al Nahda / Muhaisnah", "ratio": 180.9211491442534, "name": "RTA Taxi Rank Al Nahda Fitness Centre", "lat": 25.28258, "long": 55.372335}, {"region": "Bur Dubai", "ratio": 215.33784860557685, "name": "RTA Taxi Rank Burjman Centre A", "lat": 25.252189, "long": 55.302387}, {"region": "Bur Dubai", "ratio": 243.53685258964117, "name": "RTA Taxi Rank Burjuman Metro Station D", "lat": 25.255438, "long": 55.304969}, {"region": "Bur Dubai", "ratio": 126.46826029216463, "name": "RTA Taxi Rank ADCB Metro Station Seaside", "lat": 25.245147, "long": 55.298246}, {"region": "Bur Dubai", "ratio": 55.54349269588296, "name": "RTA Taxi Rank ADCB Metro Station Landside", "lat": 25.245274, "long": 55.298922}, {"region": "Bur Dubai", "ratio": 338.388047808764, "name": "RTA Taxi Rank Sharaf DG Metro Station B 2", "lat": 25.257434, "long": 55.296989}, {"region": "Bur Dubai", "ratio": 153.81274900398327, "name": "RTA Taxi Rank Mankhool Road Near Standard Chartered Tower", "lat": 25.25246, "long": 55.291681}, {"region": "Bur Dubai", "ratio": 69.21573705179163, "name": "RTA Taxi Rank Al Ghubaiba Bus Station Backside", "lat": 25.261755, "long": 55.288714}, {"region": "Bur Dubai", "ratio": 221.31945551128743, "name": "RTA Taxi Rank Burjuman Center B", "lat": 25.252164, "long": 55.302416}, {"region": "Bur Dubai", "ratio": 69.21573705179163, "name": "RTA Taxi Rank Al Ghubaibah Taxi Station Backside", "lat": 25.261776, "long": 55.28874}, {"region": "Bur Dubai", "ratio": 36.74415670650675, "name": "RTA Taxi Rank Ansar Gallery", "lat": 25.252236, "long": 55.309019}, {"region": "Bur Dubai", "ratio": 68.36122177954726, "name": "RTA Taxi Rank RAK Bank Near Burjuman Metro Station", "lat": 25.256451, "long": 55.305663}, {"region": "DIFC", "ratio": 497.0367170626348, "name": "RTA Taxi Rank Emirates Towers Metro Station Landside", "lat": 25.216855, "long": 55.280354}, {"region": "DIFC", "ratio": 537.9114470842328, "name": "RTA Taxi Rank Financial Centre Metro Station Seaside", "lat": 25.211724, "long": 55.274838}, {"region": "DIFC", "ratio": 698.140388768898, "name": "RTA Taxi Rank Financial Centre Metro Station Landside", "lat": 25.211297, "long": 55.276043}, {"region": "DIFC", "ratio": 412.01727861770945, "name": "RTA Taxi Rank Shangri-La Hotel", "lat": 25.208492, "long": 55.272459}, {"region": "DIFC", "ratio": 166.76889848812013, "name": "RTA Taxi Rank Dusit Thani Hotel", "lat": 25.206799, "long": 55.272998}, {"region": "DIFC", "ratio": 328.63282937365005, "name": "RTA Taxi Rank Carlton Downtown Hotel", "lat": 25.209235, "long": 55.274458}, {"region": "DIFC", "ratio": 340.0777537796964, "name": "RTA Taxi Rank Grand Stay Hotel", "lat": 25.210024, "long": 55.274986}, {"region": "DIFC", "ratio": 240.34341252699784, "name": "RTA Taxi Rank Financial Centre Metro Station B", "lat": 25.210274, "long": 55.276242}, {"region": "DIFC", "ratio": 830.5745140388764, "name": "RTA Taxi Rank Rose Rayhaan By Rotana Hotel", "lat": 25.21207, "long": 55.276378}, {"region": "DIFC", "ratio": 706.3153347732176, "name": "RTA Taxi Rank Gevora Hotel", "lat": 25.212538, "long": 55.276713}, {"region": "Deira", "ratio": 22.46810670347836, "name": "RTA Taxi Rank - Al Mamzar, Century Mall", "lat": 25.289638, "long": 55.346959}, {"region": "Deira", "ratio": 66.88180600105454, "name": "RTA Taxi Rank Hor Al Anz East, Abu Hail Center", "lat": 25.278079, "long": 55.346062}, {"region": "Deira", "ratio": 7.3151975313646584, "name": "RTA Taxi Rank Al Nahda -1, Al Mulla Plaza", "lat": 25.281348, "long": 55.357134}, {"region": "Deira", "ratio": 108.68293475171524, "name": "RTA Taxi Rank Al Muraqqbat, Al Ghurair City A", "lat": 25.266924, "long": 55.317057}, {"region": "Deira", "ratio": 60.089122579072985, "name": "RTA Taxi Rank Al Muraqqbat, Al Ghurair City B", "lat": 25.268534, "long": 55.317864}, {"region": "Deira", "ratio": 24.03564903162894, "name": "RTA Taxi Rank Riggat Al Buteen, Hilton Dubai Creek", "lat": 25.259693, "long": 55.318158}, {"region": "Deira", "ratio": 154.14166226805762, "name": "RTA Taxi Rank Al Reef Mall ", "lat": 25.270147, "long": 55.322401}, {"region": "Deira", "ratio": 277.97750619188656, "name": "RTA Taxi Rank Al Rigga Metro Station A", "lat": 25.263424, "long": 55.323447}, {"region": "Deira", "ratio": 280.59007673880336, "name": "RTA Taxi Rank Union Metro Station", "lat": 25.266278, "long": 55.315183}, {"region": "Deira", "ratio": 63.74672134475596, "name": "RTA Taxi Rank Al Qiyadah Metro Station Seaside", "lat": 25.279183, "long": 55.352021}, {"region": "Deira", "ratio": 29.7833042348443, "name": "RTA Taxi Rank Abu Hail Metro Station Seaside A", "lat": 25.276212, "long": 55.34732}, {"region": "Deira", "ratio": 30.828332453610496, "name": "RTA Taxi Rank Abu Hail Metro Station B", "lat": 25.27624, "long": 55.346146}, {"region": "Deira", "ratio": 27.17073368792881, "name": "RTA Taxi Rank Abu Hail Metro Station C", "lat": 25.275307, "long": 55.347157}, {"region": "Deira", "ratio": 66.35929189167145, "name": "RTA Taxi Rank Khaleej Road, Waterfront Market", "lat": 25.291471, "long": 55.323995}, {"region": "Deira", "ratio": 39.18855820374264, "name": "RTA Taxi Rank Souq Naif", "lat": 25.271211, "long": 55.305302}, {"region": "Deira", "ratio": 52.773925047707046, "name": "RTA Taxi Rank Canadian Specialist Hospital", "lat": 25.27793672, "long": 55.34886644}, {"region": "Deira", "ratio": 0.0, "name": "RTA Taxi Rank Deira Palm, Night Market 1", "lat": 25.29166, "long": 55.311688}, {"region": "Deira", "ratio": 2.6125705469154896, "name": "RTA Taxi Rank Seniors Happiness Centre", "lat": 25.298022, "long": 55.345618}, {"region": "Deira", "ratio": 57.99906614153931, "name": "RTA Taxi Rank Claridge Hotel Dubai ", "lat": 25.270999, "long": 55.316488}, {"region": "Deira", "ratio": 83.07974339193574, "name": "RTA Taxi Rank Salah Al Din Metro Station 2", "lat": 25.270472, "long": 55.318509}, {"region": "Deira", "ratio": 46.503755735108584, "name": "RTA Taxi Rank Abu Baker Al Siddique Metro Station 1", "lat": 25.2708, "long": 55.333412}, {"region": "Downtown Dubai / Business Bay", "ratio": 124.54680975609749, "name": "RTA Taxi Rank Business Bay, Downtown Burj Dubai - Building 3", "lat": 25.189962, "long": 55.280704}, {"region": "Downtown Dubai / Business Bay", "ratio": 241.56054634146247, "name": "RTA Taxi Rank Dubai Mall", "lat": 25.197642, "long": 55.27871}, {"region": "Downtown Dubai / Business Bay", "ratio": 35.65654634146247, "name": "RTA Taxi Rank Burj Khalifa/ Dubai Mall Metro Station Landside A", "lat": 25.200089, "long": 55.268402}, {"region": "Downtown Dubai / Business Bay", "ratio": 355.56105365853625, "name": "RTA Taxi Rank Burj Khalifa/ Dubai Mall Metro Station Landside B", "lat": 25.200639, "long": 55.269566}, {"region": "Downtown Dubai / Business Bay", "ratio": 235.53408780487746, "name": "RTA Taxi Rank Business Bay Metro Station Seaside", "lat": 25.192325, "long": 55.259923}, {"region": "Downtown Dubai / Business Bay", "ratio": 251.60464390243874, "name": "RTA Taxi Rank Business Bay Metro Station Landside", "lat": 25.190709, "long": 55.260044}, {"region": "Downtown Dubai / Business Bay", "ratio": 28.62567804877999, "name": "RTA Taxi Rank Vida Downtown Dubai", "lat": 25.18993, "long": 55.27416}, {"region": "Downtown Dubai / Business Bay", "ratio": 42.68741463414624, "name": "RTA Taxi Rank Deyaar Development PJSC (Sales Center)", "lat": 25.1858492, "long": 55.26527169}, {"region": "Downtown Dubai / Business Bay", "ratio": 37.16316097560872, "name": "RTA Taxi Rank Burj Khalifa/Dubai Mall Metro Station D", "lat": 25.19961089, "long": 55.26961725}, {"region": "Downtown Dubai / Business Bay", "ratio": 6.026458536584989, "name": "RTA Taxi Rank Emarat Petrol HQ", "lat": 25.191612, "long": 55.258374}, {"region": "Dubai Marina", "ratio": 278.0168744625966, "name": "RTA Taxi Rank Opp. Amwaj Rotana ", "lat": 25.073294, "long": 55.130559}, {"region": "Dubai Marina", "ratio": 278.70845872742785, "name": "RTA Taxi Rank Sofitel", "lat": 25.075571, "long": 55.131841}, {"region": "Dubai Marina", "ratio": 221.30696474634556, "name": "RTA Taxi Rank Opp. Le Royal Meridien ", "lat": 25.090815, "long": 55.147777}, {"region": "Dubai Marina", "ratio": 122.41041487532212, "name": "RTA Taxi Rank Opp. Al Habtoor Grand Hotel", "lat": 25.085849, "long": 55.14174}, {"region": "Dubai Marina", "ratio": 145.92427987962077, "name": "RTA Taxi Rank Opp. Grand Residence A", "lat": 25.083523, "long": 55.140719}, {"region": "Dubai Marina", "ratio": 85.75644883920847, "name": "RTA Taxi Rank Opp. Grand Residence B", "lat": 25.083249, "long": 55.140057}, {"region": "Dubai Marina", "ratio": 76.07426913155601, "name": "RTA Taxi Rank Opp. Sukoon Tower", "lat": 25.078832, "long": 55.144072}, {"region": "Dubai Marina", "ratio": 22.82228073946616, "name": "RTA Taxi Rank Opp. Tram Number 6 ", "lat": 25.086546, "long": 55.149696}, {"region": "Dubai Marina", "ratio": 354.09114359415264, "name": "RTA Taxi Rank Dubai Marina Mall", "lat": 25.07643, "long": 55.140504}, {"region": "Dubai Marina", "ratio": 74.691100601891, "name": "RTA Taxi Rank Sobha Realty Metro Station Seaside", "lat": 25.081455, "long": 55.14712}, {"region": "Dubai Marina", "ratio": 224.07330180567428, "name": "RTA Taxi Rank DMCC Metro Station Seaside", "lat": 25.071525, "long": 55.137381}, {"region": "Dubai Marina", "ratio": 99.58813413585467, "name": "RTA Taxi Rank JLT Gold Tower", "lat": 25.06930493, "long": 55.14349716}, {"region": "Dubai Marina", "ratio": 112.72823516766965, "name": "RTA Taxi Rank DoubleTree By Hilton Jumeirah", "lat": 25.071991, "long": 55.127904}, {"region": "Dubai Parks", "ratio": 2706.5753968253966, "name": "RTA Taxi Rank Dubai Outlet Village", "lat": 24.914563, "long": 55.011472}, {"region": "Dubai Parks", "ratio": 255.33730158730103, "name": "RTA Taxi Rank Bollywood Parks Dubai Park & Resorts", "lat": 24.915753, "long": 55.007915}, {"region": "Dubai Parks", "ratio": 5259.9484126984125, "name": "RTA Taxi Rank Motiongate Dubai- Dubai Parks And Resorts ", "lat": 24.921158, "long": 55.008388}, {"region": "Dubai South", "ratio": 5793.118971061092, "name": "RTA Taxi Rank Al Maktoum International Airport", "lat": 24.885799, "long": 55.15677}, {"region": "Dubailand", "ratio": 126.29048086359052, "name": "RTA Taxi Rank Arabian Ranches Community Center", "lat": 25.05711, "long": 55.271085}, {"region": "Dubailand", "ratio": 3401.4236179260715, "name": "RTA Taxi Rank Global Village", "lat": 25.072645, "long": 55.306568}, {"region": "International City / Warqa", "ratio": 48.16570261993798, "name": "RTA Taxi Rank Al Warqa -1, Al Mass Supermarket", "lat": 25.193895, "long": 55.406168}, {"region": "International City / Warqa", "ratio": 153.2545083361683, "name": "RTA Taxi Rank Dragon Mart", "lat": 25.17376, "long": 55.413906}, {"region": "Internet City", "ratio": 171.670204479064, "name": "RTA Taxi Rank Emirates Hills Second (Emaar Buildings) Emaar Business Park Buildings", "lat": 25.095254, "long": 55.166918}, {"region": "Internet City", "ratio": 154.1272638753647, "name": "RTA Taxi Rank Dubai Internet City Metro Station Landside", "lat": 25.099459, "long": 55.172341}, {"region": "Internet City", "ratio": 788.1792599805252, "name": "RTA Taxi Rank Dubai Internet City Metro Station Seaside", "lat": 25.101936, "long": 55.171947}, {"region": "Internet City", "ratio": 184.20087633885072, "name": "RTA Taxi Rank Al Khail Metro Station Seaside", "lat": 25.089324, "long": 55.156815}, {"region": "Internet City", "ratio": 12.530671859785421, "name": "RTA Taxi Rank Al Khail Metro Station Landside", "lat": 25.089288, "long": 55.158908}, {"region": "Internet City", "ratio": 125.3067185978568, "name": "RTA Taxi Rank King Salman Bin Abdulaziz Al Soud St.", "lat": 25.094947, "long": 55.154394}, {"region": "Internet City", "ratio": 66.41256085686415, "name": "RTA Taxi Rank Al Thaniya Street Barsha Heights", "lat": 25.095731, "long": 55.177456}, {"region": "Internet City", "ratio": 319.5321324245366, "name": "RTA Taxi Rank Gloria Hotel ", "lat": 25.103122, "long": 55.17139}, {"region": "Internet City", "ratio": 301.9891918208373, "name": "RTA Taxi Rank Grand Millennium Hotel Dubai", "lat": 25.101627, "long": 55.177249}, {"region": "Internet City", "ratio": 213.02142161635734, "name": "RTA Taxi Rank Internet City Al Jaddi Street", "lat": 25.101013, "long": 55.16942}, {"region": "Jebel Ali", "ratio": 1150.4958837149466, "name": "RTA Taxi Rank Ibn Batutta Mall Metro Station", "lat": 25.045643, "long": 55.117669}, {"region": "Jebel Ali", "ratio": 1.655389760740649, "name": "RTA Taxi Rank Energy Metro Station Seaside", "lat": 25.026041, "long": 55.09971}, {"region": "Jebel Ali", "ratio": 8.276948803704531, "name": "RTA Taxi Rank Energy Metro Station Landside", "lat": 25.025308, "long": 55.101136}, {"region": "Jebel Ali", "ratio": 203.61294057113454, "name": "RTA Taxi Rank Danube Metro Station Landside", "lat": 25.000319, "long": 55.09595}, {"region": "Jebel Ali", "ratio": 19.86467712889036, "name": "RTA Taxi Rank UAE Exchange Metro Station, Seaside  (Inside Jebel Ali Port)", "lat": 24.97767, "long": 55.088664}, {"region": "Jebel Ali", "ratio": 122.4988422948286, "name": "RTA Taxi Rank UAE Exchange, Landside", "lat": 24.976919, "long": 55.09116}, {"region": "Jebel Ali", "ratio": 372.46269616670907, "name": "RTA Taxi Rank Discovery Garden 1", "lat": 25.044048, "long": 55.134959}, {"region": "Jebel Ali", "ratio": 76.14792899408272, "name": "RTA Taxi Rank Discovery Garden 2", "lat": 25.044164, "long": 55.133608}, {"region": "Jebel Ali", "ratio": 342.6656804733723, "name": "RTA Taxi Rank Discovery Garden 3", "lat": 25.035241, "long": 55.145644}, {"region": "Jebel Ali", "ratio": 28.141625932594895, "name": "RTA Taxi Rank Discovery Garden 4", "lat": 25.034153, "long": 55.146393}, {"region": "Jebel Ali", "ratio": 173.81592487779773, "name": "RTA Taxi Rank Discovery Garden 5", "lat": 25.030705, "long": 55.152477}, {"region": "Jebel Ali", "ratio": 97.66799588371373, "name": "RTA Taxi Rank Discovery Garden 6", "lat": 25.030818, "long": 55.151135}, {"region": "Jebel Ali", "ratio": 779.6885773089782, "name": "RTA Taxi Rank DIP Metro Station Gate2A", "lat": 25.005012, "long": 55.155437}, {"region": "Jebel Ali", "ratio": 21.52006688963101, "name": "RTA Taxi Rank Discovery Gardens", "lat": 25.04911355, "long": 55.12793376}, {"region": "Jebel Ali", "ratio": 420.4689992281956, "name": "RTA Taxi Rank DIP Metro Station Gate 2B", "lat": 25.004211, "long": 55.155189}, {"region": "Jebel Ali", "ratio": 402.2597118600459, "name": "RTA Taxi Rank DIP Metro Station Gate 2A", "lat": 25.006111, "long": 55.15568}, {"region": "Jebel Ali", "ratio": 382.3950347311542, "name": "RTA Taxi Rank DIP Metro Station Gate 1", "lat": 25.0041, "long": 55.155572}, {"region": "Jebel Ali", "ratio": 51.31708258296784, "name": "RTA Taxi Rank Jebel Ali Metro Station Landside", "lat": 25.057292, "long": 55.127001}, {"region": "Jumeirah", "ratio": 112.4912587412582, "name": "RTA Taxi Rank Jumeirah - 3, Union Co-op Society", "lat": 25.187211, "long": 55.238977}, {"region": "Jumeirah", "ratio": 832.4353146853136, "name": "RTA Taxi Rank Mercato Shopping Centre", "lat": 25.217066, "long": 55.252515}, {"region": "Jumeirah", "ratio": 973.0493881118879, "name": "RTA Taxi Rank Four Seasons Resort Dubai At Jumeirah Beav", "lat": 25.202235, "long": 55.240601}, {"region": "Mirdif", "ratio": 27.615879828325973, "name": "RTA Taxi Rank Mirdif, West Zone", "lat": 25.231387, "long": 55.418266}, {"region": "Mirdif", "ratio": 1656.95278969957, "name": "RTA Taxi Rank City Centre Mirdif", "lat": 25.215272, "long": 55.409446}, {"region": "Mirdif", "ratio": 234.73497854077144, "name": "RTA Taxi Rank Ghoroob, Mirdif", "lat": 25.211504, "long": 55.417871}, {"region": "Mirdif", "ratio": 759.4366952789688, "name": "RTA Taxi Rank Mirdif, Uptown Shopping Centre", "lat": 25.224303, "long": 55.424004}, {"region": "Palm Jumeirah", "ratio": 50.6040532365392, "name": "RTA Taxi Rank The Palm Jumeirah", "lat": 25.116871, "long": 55.136336}, {"region": "WTC", "ratio": 215.47784647089057, "name": "RTA Taxi Rank Max Metro Station Seaside", "lat": 25.233136, "long": 55.290629}, {"region": "WTC", "ratio": 29.835394126738272, "name": "RTA Taxi Rank Max Metro Station Landside B", "lat": 25.231416, "long": 55.292624}, {"region": "WTC", "ratio": 235.36810922204938, "name": "RTA Taxi Rank World Trade Centre Metro Station Seaside", "lat": 25.224631, "long": 55.283507}, {"region": "WTC", "ratio": 44.75309119010806, "name": "RTA Taxi Rank World Trade Centre Metro Station Landside", "lat": 25.225808, "long": 55.285875}, {"region": "WTC", "ratio": 820.473338485316, "name": "RTA Taxi Rank Sheraton Grand Hotel", "lat": 25.22940049, "long": 55.2869376}, {"region": "WTC", "ratio": 391.1751674394634, "name": "RTA Taxi Rank Dubai Frame Zabeel Park", "lat": 25.234252, "long": 55.301575}, {"region": "WTC", "ratio": 900.0343894899528, "name": "RTA Taxi Rank Duja Tower ", "lat": 25.229243, "long": 55.286041}, {"region": "WTC", "ratio": 576.8176197836157, "name": "RTA Taxi Rank Commercial Tower", "lat": 25.221078, "long": 55.28105}]

new_taxi_ranks = [
    {"name": "New Taxi Rank 1", "lat": 25.180466256431913, "long": 55.266468826631275},
    {"name": "New Taxi Rank 2", "lat": 24.99495197798089, "long": 55.171474713751564},
    {"name": "New Taxi Rank 3", "lat": 25.112882226903814, "long": 55.38266977534105},
    {"name": "New Taxi Rank 4", "lat": 25.019748199057712, "long": 55.25936089340401},
    {"name": "New Taxi Rank 5", "lat": 25.129811979359044, "long": 55.11981743454644},
    {"name": "New Taxi Rank 6", "lat": 25.249541, "long": 55.480847999999995},
    {"name": "New Taxi Rank 7", "lat": 25.004931413533832, "long": 55.13292257894738},
    {"name": "New Taxi Rank 8", "lat": 25.205566610429447, "long": 55.34548024539877},
    {"name": "New Taxi Rank 9", "lat": 25.027749492063492, "long": 55.1087730952381},
    {"name": "New Taxi Rank 10", "lat": 25.15342155243446, "long": 55.294383303370786},
    {"name": "New Taxi Rank 11", "lat": 25.00711623888889, "long": 55.29499908888889},
    {"name": "New Taxi Rank 12", "lat": 24.989735794195248, "long": 55.38438055145118},
    {"name": "New Taxi Rank 13", "lat": 24.940445166666667, "long": 55.058769833333336},
    {"name": "New Taxi Rank 14", "lat": 25.09027714583333, "long": 55.31309561718749},
    {"name": "New Taxi Rank 15", "lat": 25.12728026510989, "long": 55.41935188598901},
    {"name": "New Taxi Rank 16", "lat": 25.23389803125, "long": 55.47415421875},
    {"name": "New Taxi Rank 17", "lat": 25.248527701754387, "long": 55.45693294736843},
    {"name": "New Taxi Rank 18", "lat": 24.99648933333333, "long": 55.14900333333333},
    {"name": "New Taxi Rank 19", "lat": 25.271365352941174, "long": 55.454277911764706},
    {"name": "New Taxi Rank 20", "lat": 24.963052842281883, "long": 55.14564270469799},
    {"name": "New Taxi Rank 21", "lat": 25.1293975, "long": 55.293794625000004},
    {"name": "New Taxi Rank 22", "lat": 24.946222734463273, "long": 55.216769361581925},
    {"name": "New Taxi Rank 23", "lat": 25.145768548387093, "long": 55.471818709677414},
    {"name": "New Taxi Rank 24", "lat": 25.004961215909088, "long": 55.10816451136363},
    {"name": "New Taxi Rank 25", "lat": 25.1445192962963, "long": 55.34710888888888},
    {"name": "New Taxi Rank 26", "lat": 25.26476629824561, "long": 55.26583582456141},
    {"name": "New Taxi Rank 27", "lat": 24.949926277777777, "long": 55.079474111111104},
    {"name": "New Taxi Rank 28", "lat": 25.160573624999998, "long": 55.32426732954546},
    {"name": "New Taxi Rank 29", "lat": 25.098167503703706, "long": 55.12436134814815},
    {"name": "New Taxi Rank 30", "lat": 25.042881100000006, "long": 55.4221674},
    {"name": "New Taxi Rank 31", "lat": 25.12772465100671, "long": 55.1534561901566},
    {"name": "New Taxi Rank 32", "lat": 24.865713035714286, "long": 55.14568846428572},
    {"name": "New Taxi Rank 33", "lat": 25.034575750000002, "long": 55.118469625},
    {"name": "New Taxi Rank 34", "lat": 25.0527478125, "long": 55.2312946875},
    {"name": "New Taxi Rank 35", "lat": 25.14273755555556, "long": 55.31476030555555},
    {"name": "New Taxi Rank 36", "lat": 25.24310127777778, "long": 55.398428333333335},
    {"name": "New Taxi Rank 37", "lat": 25.148011954545453, "long": 55.45705909090909},
    {"name": "New Taxi Rank 38", "lat": 24.96938, "long": 55.05678628571429},
    {"name": "New Taxi Rank 39", "lat": 24.986245894736847, "long": 55.11456507894736},
    {"name": "New Taxi Rank 40", "lat": 24.984730676470587, "long": 55.13623685294118},
    {"name": "New Taxi Rank 41", "lat": 25.017992976190477, "long": 55.16385669047619},
    {"name": "New Taxi Rank 42", "lat": 25.300349114624506, "long": 55.30564604347826},
    {"name": "New Taxi Rank 43", "lat": 24.871236181818187, "long": 55.17319981818182},
    {"name": "New Taxi Rank 44", "lat": 25.22184478571429, "long": 55.50664642857142},
    {"name": "New Taxi Rank 45", "lat": 25.191275944444445, "long": 55.31023733333333},
    {"name": "New Taxi Rank 46", "lat": 25.136127043478258, "long": 55.37819134782608},
    {"name": "New Taxi Rank 47", "lat": 24.878925000000002, "long": 55.152581222222224},
    {"name": "New Taxi Rank 48", "lat": 25.244814666666667, "long": 55.48629300000001},
    {"name": "New Taxi Rank 49", "lat": 24.855614272727276, "long": 55.15809695454546},
    {"name": "New Taxi Rank 50", "lat": 24.987033692307694, "long": 55.02310284615385},
    {"name": "New Taxi Rank 51", "lat": 25.232983777777775, "long": 55.557577333333334},
    {"name": "New Taxi Rank 52", "lat": 25.218750399999998, "long": 55.465042399999994},
    {"name": "New Taxi Rank 53", "lat": 25.201784555555555, "long": 55.30953911111111},
    {"name": "New Taxi Rank 54", "lat": 25.062476911111123, "long": 55.32010355555556},
    {"name": "New Taxi Rank 55", "lat": 24.976845733333334, "long": 55.151498066666676},
    {"name": "New Taxi Rank 56", "lat": 25.12342437857143, "long": 55.351381214285716},
    {"name": "New Taxi Rank 57", "lat": 25.150672428571426, "long": 55.41920957142857},
    {"name": "New Taxi Rank 58", "lat": 25.100550352941177, "long": 55.34047644117646},
    {"name": "New Taxi Rank 59", "lat": 25.112149916666667, "long": 55.41183158333333},
    {"name": "New Taxi Rank 60", "lat": 25.250975999999998, "long": 55.4952184},
    {"name": "New Taxi Rank 61", "lat": 25.13956629411765, "long": 55.368103999999995},
    {"name": "New Taxi Rank 62", "lat": 25.163058111111113, "long": 55.47042177777777},
    {"name": "New Taxi Rank 63", "lat": 25.165422999999997, "long": 55.44649427272727},
    {"name": "New Taxi Rank 64", "lat": 25.03545281818182, "long": 55.098425772727275},
    {"name": "New Taxi Rank 65", "lat": 25.093236533333336, "long": 55.40723377777778},
    {"name": "New Taxi Rank 66", "lat": 24.966031711538466, "long": 55.07823110576923},
    {"name": "New Taxi Rank 67", "lat": 24.873050545454547, "long": 55.04533327272727},
    {"name": "New Taxi Rank 68", "lat": 25.177360075471697, "long": 55.447909232704404},
    {"name": "New Taxi Rank 69", "lat": 25.034324692307692, "long": 55.24790992307692},
    {"name": "New Taxi Rank 70", "lat": 24.950195444444446, "long": 55.049458666666666},
    {"name": "New Taxi Rank 71", "lat": 24.816410916666666, "long": 55.23113333333334},
    {"name": "New Taxi Rank 72", "lat": 25.26970068181818, "long": 55.269392909090904},
    {"name": "New Taxi Rank 73", "lat": 24.8810626, "long": 55.1234154},
    {"name": "New Taxi Rank 74", "lat": 25.123724818181817, "long": 55.27570054545454},
    {"name": "New Taxi Rank 75", "lat": 24.996717812500002, "long": 55.14088925},
    {"name": "New Taxi Rank 76", "lat": 24.861246255813956, "long": 55.06525576744187},
    {"name": "New Taxi Rank 77", "lat": 25.0414285, "long": 55.098413333333326},
    {"name": "New Taxi Rank 78", "lat": 24.909172125000005, "long": 55.117487749999995},
    {"name": "New Taxi Rank 79", "lat": 25.14813116666667, "long": 55.43249600000001},
    {"name": "New Taxi Rank 80", "lat": 24.959801777777777, "long": 55.01160766666666},
    {"name": "New Taxi Rank 81", "lat": 25.10511525142857, "long": 55.39845161142858},
    {"name": "New Taxi Rank 82", "lat": 25.009715028846152, "long": 55.088771},
    {"name": "New Taxi Rank 83", "lat": 25.209034875, "long": 55.36199966666666},
    {"name": "New Taxi Rank 84", "lat": 25.08116711111111, "long": 55.33080527777778},
    {"name": "New Taxi Rank 85", "lat": 24.842677125, "long": 55.0243375},
    {"name": "New Taxi Rank 86", "lat": 24.980619275862068, "long": 55.02664196551724},
    {"name": "New Taxi Rank 87", "lat": 24.987112166666666, "long": 55.081466},
    {"name": "New Taxi Rank 88", "lat": 25.245620000000002, "long": 55.39047766666667},
    {"name": "New Taxi Rank 89", "lat": 25.260882428571424, "long": 55.48026257142856},
    {"name": "New Taxi Rank 90", "lat": 25.083593090909094, "long": 55.4008959090909},
    {"name": "New Taxi Rank 91", "lat": 25.160757375, "long": 55.462657625},
    {"name": "New Taxi Rank 92", "lat": 25.24910242857143, "long": 55.44119442857143},
    {"name": "New Taxi Rank 93", "lat": 24.888162, "long": 55.0677225},
    {"name": "New Taxi Rank 94", "lat": 25.154839606060605, "long": 55.511899121212124},
    {"name": "New Taxi Rank 95", "lat": 25.0512998, "long": 55.3193034},
    {"name": "New Taxi Rank 96", "lat": 24.99837542857143, "long": 55.10002328571429},
    {"name": "New Taxi Rank 97", "lat": 24.972245499999996, "long": 55.13068266666667},
    {"name": "New Taxi Rank 98", "lat": 25.224288625, "long": 55.479200375},
    {"name": "New Taxi Rank 99", "lat": 25.166559499999998, "long": 55.312820124999995},
    {"name": "New Taxi Rank 100", "lat": 24.930933666666668, "long": 55.057819},
    {"name": "New Taxi Rank 101", "lat": 25.150873333333333, "long": 55.50358049999999},
    {"name": "New Taxi Rank 102", "lat": 25.110840782608694, "long": 55.392845782608696},
    {"name": "New Taxi Rank 103", "lat": 24.995213200000002, "long": 55.092143199999995},
    {"name": "New Taxi Rank 104", "lat": 25.115129571428575, "long": 55.11647214285715},
    {"name": "New Taxi Rank 105", "lat": 25.105964529411764, "long": 55.4148955882353},
    {"name": "New Taxi Rank 106", "lat": 25.251102666666668, "long": 55.39609633333333},
    {"name": "New Taxi Rank 107", "lat": 25.221083909090908, "long": 55.343234},
    {"name": "New Taxi Rank 108", "lat": 24.960336, "long": 55.15606600000001},
    {"name": "New Taxi Rank 109", "lat": 24.9814334, "long": 55.0786456},
    {"name": "New Taxi Rank 110", "lat": 24.907399333333334, "long": 55.11089855555556},
    {"name": "New Taxi Rank 111", "lat": 25.09926466666667, "long": 55.39772333333334},
    {"name": "New Taxi Rank 112", "lat": 25.02843128571428, "long": 55.12090928571428},
    {"name": "New Taxi Rank 113", "lat": 25.2357718, "long": 55.53657059999999},
    {"name": "New Taxi Rank 114", "lat": 24.956298999999998, "long": 55.0609133888889},
    {"name": "New Taxi Rank 115", "lat": 24.968764800000002, "long": 55.1800344},
    {"name": "New Taxi Rank 116", "lat": 24.9748502, "long": 55.0705612},
    {"name": "New Taxi Rank 117", "lat": 24.9823818, "long": 55.1465974},
    {"name": "New Taxi Rank 118", "lat": 25.04035983333333, "long": 55.10787383333334},
    {"name": "New Taxi Rank 119", "lat": 24.946379800000003, "long": 55.072162000000006},
    {"name": "New Taxi Rank 120", "lat": 24.958390999999995, "long": 55.034842499999996},
    {"name": "New Taxi Rank 121", "lat": 24.9612244, "long": 55.071758},
    {"name": "New Taxi Rank 122", "lat": 25.308271800000004, "long": 55.35288920000001}
]


# Configure Gemini API
genai.configure(api_key="AIzaSyCokG5u0H9uADiOfUpSfrMlET4bawr8GIk")

@app.route('/send-to-gemini', methods=['POST'])
def send_to_gemini():
    try:
        predictions = request.json.get('predictions', [])

        # Format prediction text
        formatted_text = "\n".join([f"{p['name']}: {p['value']}%" for p in predictions])

        # Query Gemini
        response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
            f"Summarize the following taxi demand predictions:\n{formatted_text}"
        )

        # Extract response
        summary = response.text if hasattr(response, 'text') else "No summary available."

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)})

# Function to generate 30-minute time intervals
def generate_time_intervals():
    return [(hour, minute) for hour in range(24) for minute in [0, 30]]

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_time = None
    if request.method == 'POST':
        selected_time = request.form['time']
    return render_template('index.html', selected_time=selected_time)

@app.route('/new-ranks')
def new_ranks():
    return render_template('new-ranks.html')

@app.route('/rank-bar-chart')
def rank_bar_chart():
    return render_template('rank-bar-chart.html')

@app.route('/dubai-map')
def dubai_map():
    dubai_coordinates = [ 25.1513 , 55.2410]
    folium_map = folium.Map(
        location=dubai_coordinates, 
    zoom_start=11, 
    tiles="OpenStreetMap", 
    scrollWheelZoom=False,
    height=500,
    )
    
    # Add custom CSS to remove scrollbars
    folium_map.get_root().header.add_child(folium.Element("""
        <style>
            .folium-map::-webkit-scrollbar { display: none; }
            .leaflet-container { overflow: hidden !important; }
            ::-webkit-scrollbar { display: none; }
            * { scrollbar-width: none; }
            #map { overflow: hidden !important; }
        </style>
    """))

    for rank in taxi_ranks:
        folium.Marker(
            location=[rank["lat"], rank["long"]],
            popup=rank["name"],
            icon=folium.Icon(color="blue", icon="taxi", prefix="fa")
        ).add_to(folium_map)

    return folium_map._repr_html_()

@app.route('/taxi-rank-map')
def taxi_rank_map():
    dubai_coordinates = [25.1513 , 55.2410]
    folium_map = folium.Map(
        location=dubai_coordinates, 
        zoom_start=11, 
        tiles="OpenStreetMap", 
        scrollWheelZoom=False,
        height=645  # Set fixed height
    )
    
    # Add custom CSS to remove scrollbars
    folium_map.get_root().header.add_child(folium.Element("""
        <style>
            .folium-map::-webkit-scrollbar { display: none; }
            .leaflet-container { overflow: hidden !important; }
            ::-webkit-scrollbar { display: none; }
            * { scrollbar-width: none; }
            #map { overflow: hidden !important; }
        </style>
    """))

    # Add existing taxi ranks
    for rank in taxi_ranks:
        folium.Marker(
            location=[rank["lat"], rank["long"]],
            popup=rank["name"],
            icon=folium.Icon(color="blue", icon="taxi", prefix="fa")
        ).add_to(folium_map)

    # Add new taxi ranks
    for rank in new_taxi_ranks:
        folium.Marker(
            location=[rank["lat"], rank["long"]],
            popup=rank["name"],
            icon=folium.Icon(color="orange", icon="taxi", prefix="fa")
        ).add_to(folium_map)

    return folium_map._repr_html_()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json().get('time', 0)
        input_data = int(input_data)

        time_for_prediction = pd.DataFrame({'Hour': [input_data]})
        predicted_region_proba = model.predict_proba(time_for_prediction)[0]
        predicted = predicted_region_proba * 100

        # Normalize probabilities
        min_proba = np.min(predicted_region_proba)
        max_proba = np.max(predicted_region_proba)
        normalized_proba = (predicted_region_proba - min_proba) / (max_proba - min_proba + 1e-6)

        # Create map with consistent height
        dubai_coordinates = [25.1513 , 55.2410]
        folium_map = folium.Map(
            location=dubai_coordinates, 
            zoom_start=11, 
            tiles="OpenStreetMap", 
            scrollWheelZoom=False,
            height=530  # Set fixed height
        )
        
        # Add custom CSS to remove scrollbars
        folium_map.get_root().header.add_child(folium.Element("""
        <style>
            .folium-map::-webkit-scrollbar { display: none; }
            .leaflet-container { overflow: hidden !important; }
            ::-webkit-scrollbar { display: none; }
            * { scrollbar-width: none; }
            #map { overflow: hidden !important; }
        </style>
        """))

        # Add regions with color gradients
        for i, region in enumerate(regions_data):
            normalized_probability = normalized_proba[i]
            color = mcolors.to_hex(mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])(normalized_probability))
            
            folium.Circle(
                location=[region["lat"], region["long"]],
                radius=region["radius"] * 1000,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"{region['name']}: {predicted[i]:.2f}% demand"
            ).add_to(folium_map)

        # Label mapping for regions
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

        # Add taxi rank markers with updated ratios
        for rank in taxi_ranks:
            for idx, region in enumerate(label_mapping.values()):
                if rank["region"] == region:
                    updated_ratio = rank["ratio"] * (predicted_region_proba[idx])
                    updated_ratio = int(np.ceil(float(updated_ratio)))

                    folium.Marker(
                        location=[rank["lat"], rank["long"]],
                        popup=f"{rank['name']}<br><br>Prediction: {updated_ratio} Taxi(s)",
                        icon=folium.Icon(color="blue", icon="taxi", prefix="fa")
                    ).add_to(folium_map)
                    break

        return folium_map._repr_html_()

    except Exception as e:
        print(f"Error while making predictions: {e}")
        return jsonify({"error": str(e)})
    
@app.route('/demand-chart')
def demand_chart():
    return render_template('chart.html')

@app.route('/bar-chart')
def bar_chart():
    return render_template('bar-chart.html')

@app.route('/update-chart', methods=['POST'])
def update_chart():
    try:
        input_data = request.get_json().get('time', 0)
        input_data = int(input_data)

        time_for_prediction = pd.DataFrame({'Hour': [input_data]})
        predicted_region_proba = model.predict_proba(time_for_prediction)[0]
        predicted = predicted_region_proba * 100

        # Create array of predictions with region names
        predictions = []
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

        predictions = [
            {
                "name": label_mapping[i],
                "value": float(predicted[i])
            }
            for i in range(len(predicted))
        ]

        # Calculate taxi ranks data
        ranks_data = []
        for rank in taxi_ranks:
            for idx, region in enumerate(label_mapping.values()):
                if rank["region"] == region:
                    # Calculate number of taxis for this rank
                    taxis = int(np.ceil(float(rank["ratio"] * predicted_region_proba[idx])))
                    ranks_data.append({
                        "name": rank["name"],
                        "taxis": taxis
                    })
                    break

        # Return both predictions and ranks data
        return jsonify({
            "predictions": predictions,
            "ranks": ranks_data
        })

    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)