from multiprocessing.resource_sharer import stop
from tokenize import Double
from flask import Flask, redirect, url_for, render_template, request
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import jellyfish

app = Flask(__name__)

@app.route('/')
#This function loads the main page of the GUI
def index():
    return render_template('cw_detect_gui.html')
    

@app.route("/", methods=["POST", "GET"])
#This function accepts POST methods from the GUI, processes it and returns it
#If the GUI sends a GET method, it just returns to the main page of the GUI
def string_input():
    if request.method == "POST":
        str_input = request.form.get('input_sentence')
        str_input_lower = str_input.lower()
        str_token = string_tokeniser(str_input_lower)
        str_no_sw = remove_stopwords(str_token)
        matches = cw_detection(str_no_sw)
        match_summary = freq_counter(matches[0], matches[1])
        return render_template('cw_detect_gui.html', input_sentence=str_input, 
        sasa_freq=(match_summary[0] + "(" + match_summary[8] + ")"),
        be_freq=(match_summary[1] + "(" + match_summary[9] + ")"),
        mpa_freq=(match_summary[2] + "(" + match_summary[10] + ")"),
        sog_freq=(match_summary[3] + "(" + match_summary[11] + ")"),
        r_freq=(match_summary[4] + "(" + match_summary[12] + ")"),
        er_freq=(match_summary[5] + "(" + match_summary[13] + ")"),
        c_freq=(match_summary[6] + "(" + match_summary[14] + ")"),
        o_freq=(match_summary[7] + "(" + match_summary[15] + ")"),
        freq_total=(match_summary[16] + "(" + match_summary[17] + ")"),
        direct_matches=matches[0],
        indirect_matches=matches[1])
        
    else:
        return render_template('cw_detect_gui.html')

#This function tokenises a string seperated by " "(Space) into a list of words and returns it
def string_tokeniser(input_str):
    words = word_tokenize(input_str)
    return words

#This function removes stop words from a list 
def remove_stopwords(token_str):
    stop_words = set(stopwords.words('english'))
    no_sw_str = [w for w in token_str if not w in stop_words]
    no_sw_str = []
    for w in token_str:
        if w not in stop_words:
            no_sw_str.append(w)
    
    return no_sw_str

#This function calculates and summarises the results of the curse word detection process and returns it
def freq_counter(direct_matches, indirect_matches):
    sasa_freq:int = 0
    be_freq:int = 0
    mpa_freq:int = 0
    sog_freq:int = 0
    r_freq:int = 0
    er_freq:int = 0
    c_freq = 0
    o_freq = 0

    in_sasa_freq:int = 0
    in_be_freq:int = 0
    in_mpa_freq:int = 0
    in_sog_freq:int = 0
    in_r_freq:int = 0
    in_er_freq:int = 0
    in_c_freq = 0
    in_o_freq = 0

    for matches in direct_matches:
        match matches[2]:
            case "Sexual Anatomy/Sexual Acts":
                sasa_freq += 1
            case "Bodily Excretions":
                be_freq += 1
            case "Mental/Physical Attributes":
                mpa_freq += 1
            case "Sexual Orientation/Gender":
                sog_freq += 1
            case "Religion":
                r_freq += 1
            case "Ethnicity/Race":
                er_freq += 1
            case "Class":
                c_freq += 1
            case "Other":
                o_freq += 1               

    for matches in indirect_matches:
        match matches[2]:
            case "Sexual Anatomy/Sexual Acts":
                in_sasa_freq += 1
            case "Bodily Excretions":
                in_be_freq += 1
            case "Mental/Physical Attributes":
                in_mpa_freq += 1
            case "Sexual Orientation/Gender":
                in_sog_freq += 1
            case "Religion":
                in_r_freq += 1
            case "Ethnicity/Race":
                in_er_freq += 1
            case "Class":
                in_c_freq += 1
            case "Other":
                in_o_freq += 1
        
    temp_match_summary = [sasa_freq,be_freq,mpa_freq,sog_freq,r_freq,er_freq,c_freq,o_freq,in_sasa_freq,in_be_freq,in_mpa_freq,in_sog_freq,in_r_freq,in_er_freq,in_c_freq,in_o_freq]

    total_direct_matches = sum(temp_match_summary[0:8])
    temp_match_summary.append(total_direct_matches)

    total_indirect_matches = sum(temp_match_summary[8:16])
    temp_match_summary.append(total_indirect_matches)

    match_summary = list(map(str, temp_match_summary))

    return match_summary

#This function compares the input parameter list with the curse word dictionary list and returns a list of direct and indirect matches alongside their "distances"
def cw_detection(no_sw_str):
    
    curse_word_dictionary = read_cw_dict()
    direct_matches = []
    indirect_matches = []

    for word in no_sw_str:
        for row in curse_word_dictionary:
            dam_lev_dis:int = jellyfish.damerau_levenshtein_distance(word, row[0])
            jaro_dis:Double = jellyfish.jaro_distance(word, row[0])
            if (word == row[0]):
                direct_matches.append((word, row[0], row[1], dam_lev_dis, jaro_dis))
            elif (dam_lev_dis <= 4) and (jaro_dis >= 0.90):
                indirect_matches.append((word, row[0], row[1], dam_lev_dis, jaro_dis))
            
    for row2 in direct_matches:
        for row1 in indirect_matches[:]:
            if row1[0] == row2[0]:
                    indirect_matches.remove(row1)     

    for row2 in indirect_matches:
        for row1 in indirect_matches[:]:
            if row1 == row2:
                pass
            elif row1[0] == row2[0]:
               indirect_matches.remove(row1)

    matches = [direct_matches,indirect_matches]

    return matches
    
#This function reads the curse word dictionary and returns a created list 
def read_cw_dict():
    with open('cw_dict.csv', encoding='utf-8') as d:
        reader = csv.reader(d)
        cw_dict_list = list(reader)
    return cw_dict_list
