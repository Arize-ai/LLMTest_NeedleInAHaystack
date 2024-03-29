import litellm
from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
from anthropic import Anthropic
import numpy as np
import pandas as pd
import random
import nest_asyncio
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phoenix.evals.models import LiteLLMModel
from phoenix.evals.models.vertex import GeminiModel
from phoenix.evals.models.anthropic import AnthropicModel
import asyncio
from phoenix.experimental.evals.utils import snap_to_rail
from phoenix.experimental.evals import (
    OpenAIModel,
    llm_generate,
)
from openai import OpenAI
from phoenix.trace.openai import OpenAIInstrumentor

# Initialize OpenAI auto-instrumentation
OpenAIInstrumentor().instrument()

from datetime import datetime, timedelta
from waffle import Waffle

#from google.cloud import aiplatform
import vertexai.preview
from transformers import T5Tokenizer
import phoenix as px

session = px.launch_app()

load_dotenv()

# Generate timeseries, for each type of generation, run a single time to get token count
# First we calculate each time series token count
# We look at total window, determine total time series we can fit
# Params are number of anomalies, number of anomaly events, anomaly type, and anomaly severity
# For each anomaly, we create an array that represents which time series in the total, will have it
# [0, 1, 0, 0, 1] would be the 2nd and the 5th time series would have an anomaly event for analomaly A
# we generate an array for each anomaly type
# We then iterate through each time series, and for each anomaly type, and generate it
# We run the model to do timeseries detection and collect the name of the time series and the anamolies deceted


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                 ###########################################
                 ###### UNCOMMMENT only 1 Provider #########
                 model_provider = "OpenAI",
                 #model_provider = "Anthropic",
                 #model_provider = "Perplexity",
                 #model_provider = "Anyscale",
                 #model_provider = "Mistral",
                 #model_provider = "LiteLLM",
                 #model_provider = "GoogleVertex",
                 #############################################
                 ###### UNCOMMMENT only 1 model name #########
                 #model_name='gpt-4-1106-preview',
                 model_name='gpt-4-0125-preview',
                 #model_name='gpt-3.5-turbo-1106',
                 #model_name='gpt-4',
                 #model_name='claude-2.1',
                 #model_name='claude-3-opus-20240229',
                 #model_name ='claude-3-sonnet-20240229',
                 #model_name='gemini-pro',
                 #model_name='gemini-pro-vision',
                 #model_name='mistral/mistral-medium',
                 #model_name='mistral/mistral-small',
                 #model_name='mistral/mistral-tiny',
                 #model_name='mistralai/Mistral-7B-Instruct-v0.1'
                 #model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'
                 #model_name='together_ai/togethercomputer/llama-2-70b-chat',
                 #model_name='huggingface/microsoft/phi-2',
                 #############################################
                 time_series_type="20_to_26", #1_to_4 20_to_26
                 num_time_series_days=30,
                anomaly_dict={"Main": 
                               {"start_date": "random", "day_length": 1, "num_of_dimensions_with_anomaly":20, 
                                "anomaly_type":"value_change_percent", "value":11 }  #value_change_percent value_change
                               }, #"anomaly_low": 0, "anomaly_high": 100, "anomaly_n_days": 1, "anomaly_percent": 0.1
                #anomaly_dict={"Main": 
                #               {"start_date": datetime.now() + timedelta(days=5), "day_length": 1, "num_of_dimensions_with_anomaly":20, 
                #                "anomaly_type":"value_change_percent", "value":11 }  #value_change_percent value_change
                #               }, #"anomaly_low": 0, "anomaly_high": 100, "anomaly_n_days": 1, "anomaly_percent": 0.1
                 format_type = "normal", #std_dev" "normal"
                 prompt_direction = "percent_move", # "std_dev", "10_point_move" "percent_move"
                 anomaly_percentage = 90,
                 prompt_percentage_detect = 40,
                 noise_level_percent = 20,
                 context_lengths_min = 30000, #Always inclusive of Min, interval = 1 wil linclude
                 context_lengths_max = 110000, #Max will get include wiht interval 2+
                 context_lengths_num_intervals = 4, #Nice balance between speed and fidelity
                 context_lengths = None,
                 document_depth_percent_min = 0,
                 document_depth_percent_max = 100,
                 #document_depth_percent_intervals = 5, #Uncomment for fast testing run
                 document_depth_percent_intervals = 30, #Nice balance between speed and fidelity
                 #document_depth_percent_intervals = 35, #Uncomment for high fidelity run
                 document_depth_percents = None,
                 results_version = 1,
                 rnd_number_digits = 2,
                 document_depth_percent_interval_type = "linear",
                 anthropic_template_version = "rev1",
                 openai_api_key=None,
                 anthropic_api_key = None,
                 save_results = False,
                 final_context_length_buffer = 200,
                 print_ongoing_status = True,
):
        """        
        :param rnd_number_digits: The number of digits in the random number. Default is 7.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        self.rnd_number_digits = rnd_number_digits
        self.time_series_type = time_series_type
        self.context_lengths_max = context_lengths_max
        self.num_time_series_days= num_time_series_days
        self.anomaly_dict = anomaly_dict
        self.anomaly_percentage = anomaly_percentage
        self.prompt_percentage_detect = prompt_percentage_detect
        self.noise_level_percent = noise_level_percent
        self.prompt_direction = prompt_direction
        self.format_type = format_type
        self.context_lengths_num_intervals = context_lengths_num_intervals
        self.document_depth_percent_intervals = document_depth_percent_intervals
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.anthropic_template_version = anthropic_template_version 
        self.testing_results = []

        print("model_provider: " + model_provider)
        print("model_name: " + model_name)

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        if model_provider not in ["OpenAI", "Anthropic", "Anyscale", "Perplexity", "GoogleVertex", "Mistral", "LiteLLM"]:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
        if model_provider == "Anthropic" and "claude" not in model_name:
            raise ValueError("If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models")
        
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name

        if model_provider == "OpenAI":
            if not self.openai_api_key and not os.getenv('OPENAI_API_KEY'):
                raise ValueError("Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model")
            else:
                self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env.")
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv('ANTHROPIC_API_KEY')
            
        if not self.model_name:
            raise ValueError("model_name must be provided.")

        if model_provider == "Anthropic":
            self.enc = Anthropic().get_tokenizer()
        elif model_provider == "OpenAI":
            self.enc = tiktoken.encoding_for_model(self.model_name)
        elif model_provider == "GoogleVertex":
            self.enc = T5Tokenizer.from_pretrained("google-t5/t5-11b")
        else:
            self.enc = tiktoken.encoding_for_model("gpt-4")

        self.google_project = os.getenv('GOOGLE_PROJECT')
        self.google_location = os.getenv('GOOGLE_LOCATION')

        if model_provider == "GoogleVertex":
            if not self.google_project:
                raise ValueError("Either google_project must be supplied with init, or GOOGLE_PROJECT must be in env.")
            if not self.google_location:
                raise ValueError("Either google_location must be supplied with init, or GOOGLE_LOCATION must be in env.")

        self.model_to_test_description = model_name
        self.directions = self.get_directions(prompt_direction = self.prompt_direction, prompt_percentage_detect=self.prompt_percentage_detect)
        return None

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        contexts = []
        #Evaluation of the model performance 
        #Uses Phoenix Evals
        if self.model_provider == "OpenAI":
            model = OpenAIModel(model_name=self.model_name,max_tokens=4096 )
        elif self.model_provider == "Anthropic":
            model = AnthropicModel(model=self.model_name, temperature=0.0,max_tokens=4096)

        elif self.model_provider == "LiteLLM":
            model = LiteLLMModel(model_name=self.model_name, temperature=0.0,max_tokens=4096)
            litellm.set_verbose=True
            litellm.vertex_project = self.google_project
            litellm.vertex_location = self.google_location

        elif self.model_provider == "GoogleVertex":
            
            #template = self.SIMPLE_TEMPLATE
            aiplatform.init(
                # your Google Cloud Project ID or number
                # environment default used is not set
                project=self.google_project,

                # the Vertex AI region you will use
                # defaults to us-central1
                location=self.google_location,)
            model = GeminiModel()
        else:
            model = LiteLLMModel(model_name=self.model_name, temperature=0.0)
            #litellm.set_verbose=True

        # DataFrame To Return
        results_df = pd.DataFrame(columns=['timeseries_data', 'directions', 'anomalies', 'number_of_timeseries', 'context_length',
                                           'depth_percent', 'normal_timeseries_slots'])
        timeseries_counts = {}
        for context_length in self.context_lengths:
            # Call the contet function
            timeseries_data, directions, anomalies, number_of_timeseries, normal_timeseries_slots = self.generate_timeseries_context(
                window_tokens=context_length,
                time_series_type=self.time_series_type,
                num_time_series_days=self.num_time_series_days,
                anomaly_dict=self.anomaly_dict,
                format_type=self.format_type,
                anomaly_percentage=self.anomaly_percentage,
                noise_level_percent=self.noise_level_percent
            )
            timeseries_counts[context_length] = number_of_timeseries
            # Append to DataFrame
            #anomalies_str = str(anomalies)  # Converting dictionary to string for simplicity; adjust as needed
            # Create a new DataFrame for the row you want to add
            new_row = pd.DataFrame({'timeseries_data': [timeseries_data], 'directions': [directions], 'anomalies': [anomalies],
                                   'number_of_timeseries': [number_of_timeseries], 
                                   'context_length':[context_length],
                                   'normal_timeseries_slots': [normal_timeseries_slots]})

            # Use concat to add the new row to the existing DataFrame
            results_df = pd.concat([results_df, new_row], ignore_index=True)    
        template = self.get_prompt_template()
        # The rails is used to search outputs for specific values and return a binary value
        # It will remove text such as ",,," or "..." and general strings from outputs
        # It answers needle_rnd_number or unanswerable or unparsable (if both or none exist in output)

        def find_anomalies_in_time_series(output, row_index):
            # This is the function that will be called for each row of the dataframe
            row = results_df.iloc[row_index]
            anomaly_data =  row["anomalies"]
            normal_timeseries_slots = row["normal_timeseries_slots"]
            total_anomalies, all_anomalies_detected, false_posiive,false_positive_slots = self.check_anomaly_data(output, anomaly_data, normal_timeseries_slots)

            # If the needle is in the output, then it is answerable
            if total_anomalies == all_anomalies_detected:
                if total_anomalies == 0:
                    print("❌ No anomalies inserted into the data")
                else:
                    print("✅ Found all anomalies!")
                    if false_posiive:
                        print("❌ Detected false positive")
            else:
                # If the needle is not in the output, then it is unanswerable
                print("---------------------------------------------------------------------")
                print(f"❌ Did not find the all anomalies {total_anomalies}, found: {all_anomalies_detected}")
                if false_posiive:
                    print("❌ Detected false positive")
                print(row)
            return {
                'label': total_anomalies == all_anomalies_detected,
                'all_anomalies_detected': all_anomalies_detected,
                'total_anomalies' : total_anomalies,
                'anomaly_data' : anomaly_data,
                'false_posiive': false_posiive,
                'false_positive_slots': false_positive_slots
            }

        #This is the core of the Phoenix evaluation
        #It runs the model on every row of the dataframe
        #It looks for columns that are defined in the template question/context
        #The generation of the model, the output, is "cleaned" up by the rails
        #The rails are used to search for specific values in the output
        #The output is then classified as either needle_rnd_number, unanswerable, or unparsable
        #This runs a number of threads in parallel speeding up the generation/Evaluation process
        nest_asyncio.apply()  # Run async
        needle_test_results = llm_generate(
            dataframe=results_df,
            template=template,
            model=model,
            verbose=True,
            concurrency=1,
            # Callback function that will be called for each row of the dataframe
            # Used to find the needle in the haystack
            output_parser=find_anomalies_in_time_series,
            # These two flags will add the prompt / response to the returned dataframe
            include_prompt=True,
            include_response=True,         
        )
        run_name = (self.model_provider + '_' + self.model_name + "_" + str(self.context_lengths_num_intervals)  + "_" + str(self.document_depth_percent_intervals) +
                     "_" + str(self.format_type) + "_" + (self.prompt_direction) + "_anom-%:" + str(self.anomaly_percentage)  ).replace("/", "_")
        df = pd.concat([results_df, needle_test_results], axis=1)
        df_final = self.prep_anomaly_data(df)
        df_final['score'] = df_final.apply(lambda row: self.check_row(row), axis=1)
        average_scores = df_final.groupby(['context_length', 'index_position'])['score'].mean().reset_index()
        average_scores['color'] = average_scores['score'].apply(self.score_to_color)
        self.generate_waffle_image(average_scores, timeseries_counts, run_name, df)
        df.to_csv("save_results_" + run_name + "_.csv")
    
        return True

    def get_directions(self, prompt_direction, prompt_percentage_detect):

        if prompt_direction == "std_dev":
            directions = ''' 
            You are looking for anomalies in the time series data. Any movement of more than 2x the standard deviation in units in a single day should be considered an anomaly.
            Each dimension has the standard deviaiton pre calculated for you, please use that.
            You are not writing code, just trying to find any dimensions that have data movements that are considered anomalies.
            If an anomaly is detected for a dimension, please return the date it occurred and the dimension name.
            If multiple dimensions have anomalies on the same day, please return the date, then list all the dimensions. 
            If no anomaly is detected, please do not mention the dimension name, only name the dimensions that have anomalies.
        '''
        elif prompt_direction == "percent_move":
            directions = ''' 
            You are looking for anomalies in the time series data. Any movement of more {percent}% in a single day should be considered an anomaly.
            You are looking for a movement in a single day where the value is {percent}% higher or lower than the previous day.
            Remember if it is not higher than {percent}% don't mention the dimension name.
            You are not writing code, just trying to find any dimensions that have data movements that are considered anomalies.
            If an anomaly is detected for a dimension, please return the date it occurred and the dimension name.
            If multiple dimensions have anomalies on the same day, please return the date, then list all the dimensions. 
            If no anomaly is detected, please do not mention the dimension name, only name the dimensions that have anomalies.
        '''.format_map({'percent': prompt_percentage_detect})

        else:
            directions = ''' 
            You are looking for anomalies in the time series data. Any movement of more than 10 units in a single day should be considered an anomaly.
            You are not writing code, just trying to find any dimensions that have data movements that are considered anomalies.
            If an anomaly is detected for a dimension, please return the date it occurred and the dimension name.
            If multiple dimensions have anomalies on the same day, please return the date, then list all the dimensions. 
            If no anomaly is detected, please do not mention the dimension name, only name the dimensions that have anomalies.
        '''
        return directions

    def get_prompt_template(self):
        basic_template = ''' You are an AI assistant for a data scientist. You have been given a time series dataset to analyze. 
        The dataset contains a series of measurements taken at regular intervals over a period of time. 
        There is one timeseries for each city in the dataset. Your task is to identify any anomalies in the data. The dataset is in the form of a JSON object,
        with the date as the key and the measurement as the value. 
        
        The dataset is as follows: 
        {timeseries_data}

        Please use the follow directions to analyze the data. 
        {directions}
        
        '''
        return basic_template

    def generate_timeseries_context(self, window_tokens, time_series_type, num_time_series_days, anomaly_dict, format_type,
                                    anomaly_percentage, noise_level_percent):
        time_series_start_date = datetime.now().strftime('%Y-%m-%d') 
        #anomaly_type, anomaly_low, anomaly_high, anomaly_n_days, anomaly_percent
        test_timeseries_df = self.generate_timeseries(time_series_type=time_series_type, num_time_series_days=num_time_series_days, 
                            time_series_start_date=time_series_start_date, noise_level_percent=noise_level_percent)
        
        test_timeseries_str = self.format_timeseries_string(test_timeseries_df, "Test Dimension", format_type=format_type)
        timeseries_token_len = self.get_context_length_in_tokens(test_timeseries_str)
        number_of_timeseries = int((window_tokens ) / timeseries_token_len)

        schedule_of_anomaly = {}
        anomaly_checks = {}
        normal_timeseries_slots = {}
        timeseries_data = ''
        #Iterate through the list of anomalies you want to create and create a schedule
        for anomaly_name, amomaly_value in anomaly_dict.items():
            #This will generate an array of IDs that represent the timeseries number
            schedule_of_anomaly[anomaly_name] = self.generate_schedule_of_anomaly(amomaly_value, number_of_timeseries)
            anomaly_checks[anomaly_name] = {} # To fill in later

        for time_series_index in range(number_of_timeseries):
            dimension_name = LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES[time_series_index]
            current_time_series = self.generate_timeseries(time_series_type=time_series_type, num_time_series_days=num_time_series_days, 
                            time_series_start_date=time_series_start_date, noise_level_percent=noise_level_percent)
            for anomaly_name, schedule in schedule_of_anomaly.items():
                #Generate the anomaly for the time series
                if time_series_index in schedule:
                    current_time_series = self.generate_anomaly(current_time_series, anomaly_metadata=anomaly_dict[anomaly_name], anomaly_percentage=anomaly_percentage)
                    token_position = self.get_context_length_in_tokens(timeseries_data)
                    if dimension_name in normal_timeseries_slots:
                        del normal_timeseries_slots[dimension_name]
                    anomaly_checks[anomaly_name][dimension_name] = {"index_position": time_series_index, 
                                                                    "token_position": token_position, 
                                                                    "depth_percent":int((token_position/window_tokens)*100),
                                                                    "anomaly_metadata": anomaly_dict[anomaly_name],
                                                                    "timeseries_df": current_time_series,}
                else:
                    normal_timeseries_slots[dimension_name] = time_series_index
            timeseries_to_add = self.format_timeseries_string(current_time_series, dimension_name,format_type=format_type)
            timeseries_data += timeseries_to_add
            
        return timeseries_data, self.directions, anomaly_checks, number_of_timeseries, normal_timeseries_slots

    #This function will check if the output labeled and dected all of the anomalies
    def check_anomaly_data(self, output, anomaly_checks, normal_timeseries_slots):
        all_anomalies_detected = 0
        total_anomalies = 0
        dimensions_with_anomalies = []
        false_positive = False
        false_positive_slots = {}
        for anomaly_name in anomaly_checks:
            for dimension_name, anomaly_data in anomaly_checks[anomaly_name].items():
                dimensions_with_anomalies.append(dimension_name)
                total_anomalies += 1
                #Check if the anomaly is in the output
                if self.check_dimension(dimension_name, output):
                    all_anomalies_detected +=1 
                    anomaly_checks[anomaly_name][dimension_name]["detected"] = True
                else:
                    print("Anomaly not detected:  " + dimension_name)
                    anomaly_checks[anomaly_name][dimension_name]["detected"] = False
        for dimension_check in  LLMNeedleHaystackTester.RANDOM_NEEDLE_CITIES:
            if not dimension_check in dimensions_with_anomalies:
                if self.check_dimension(dimension_check, output):
                    if dimension_check in normal_timeseries_slots:
                        false_positive_slots[dimension_check] = normal_timeseries_slots[dimension_check]
                        print("False Positive in data:  " + dimension_check)
                    else:
                        print("False Positive not in data:  " + dimension_check)
                    false_positive = True
        return total_anomalies, all_anomalies_detected, false_positive, false_positive_slots
        #Add checks for extrenous dimnesion mentions in output
                
    #Simple check to start 
    def check_dimension(self, dimension_name, output):
        if dimension_name.lower() in output.lower():
            return True
        else:
            return False


    def generate_timeseries(self, time_series_type, num_time_series_days, time_series_start_date,noise_level_percent):
        # Convert start date string to datetime object
        start_date = datetime.strptime(time_series_start_date, '%Y-%m-%d')
        # Create a range of dates starting from the start date
        date_range = [start_date + timedelta(days=x) for x in range(num_time_series_days)]
        noise_level = (1 + noise_level_percent/100)
        # Generate values based on the specified time series type
        if time_series_type == "1_to_4":
            values = round(random.uniform(1, 4), 2)
        elif time_series_type == "0_to_1_rnd2":
            # Generate random floats and round to 2 decimal places
            values = np.round(np.random.rand(num_time_series_days), 2)
        elif time_series_type == "20_to_26":
            min_value = 20
            values = np.round(min_value + np.random.rand(num_time_series_days) * (min_value * noise_level - min_value), 2)
        elif time_series_type == "1_to_10000":
            values = round(random.uniform(1, 10001 ), 2)
        elif time_series_type == "-10000_to_10000":
            values = round(random.uniform(-10000, 10001 ), 2)
        else:
            # Return None if an invalid type is specified
            assert False, "Invalid time series type"
        
        # Create and return the dataframe
        df = pd.DataFrame({'Date': date_range, 'Value': values})
        return df

    def format_timeseries_string(self, df, dimension_name, format_type):
        # Convert the dataframe to a dictionary with dates as keys and values as values
        # Assuming 'Date' is in a datetime format. If not, you'll need to convert it.
        data_dict = df.set_index('Date').T.to_dict('records')[0]
        data_dict = {date.strftime('%Y-%m-%d'): value for date, value in data_dict.items()}
        
        # Convert the dictionary to a JSON string
        json_string = json.dumps(data_dict)
        # Check if format_type is "std_dev" and add standard deviation to output_string
        if format_type == "std_dev":
            std_dev = df['Value'].std()
            output_string = f'"Data for dimension - {dimension_name}": \n Standard Deviation: {std_dev:.2f} \n {json_string}\n'
        else:
            # Initial output string without standard deviation
            output_string = f'"Data for dimension - {dimension_name}": \n {json_string}\n'
        
        return output_string

    
    def generate_schedule_of_anomaly(self, anomaly_value, number_of_timeseries):
        # Extract the number of anomalies from the dictionary
        num_anomalies = anomaly_value.get("num_of_dimensions_with_anomaly", 0)

        # Ensure the number of anomalies does not exceed the number of timeseries slots
        num_anomalies = min(num_anomalies, number_of_timeseries)
        
        # Generate a list of unique random indices for anomalies without replacement
        anomaly_indices = np.random.choice(number_of_timeseries, num_anomalies, replace=False)

        return anomaly_indices.tolist()

    def generate_anomaly(self, df, anomaly_metadata, anomaly_percentage):
        start_date_arg = anomaly_metadata.get("start_date")
        day_length = anomaly_metadata.get("day_length", 0)
        anomaly_type = anomaly_metadata.get("anomaly_type", 0)
        anomaly_value_change = anomaly_metadata.get("value", 0)

        # Calculate end date
        if start_date_arg == "random":
           # Exclude the first and last row for the sampling
            subset_df = df.iloc[1:-1]  # This excludes the first and last row
            start_date = pd.to_datetime(subset_df['Date'].sample().values[0])
        else:
            start_date = start_date_arg            
        
        end_date = start_date + timedelta(days=day_length)

        if anomaly_type == "value_change":
            # Apply the specified value change for the date range
            df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date), 'Value'] += anomaly_value_change
        elif anomaly_type == "value_change_percent":
            current_date = datetime.strptime(start_date, '%Y-%m-%d') if isinstance(start_date, str) else start_date
            while current_date < end_date:
                # Calculate the next day to avoid changing the first day selected
                next_day = (current_date + timedelta(days=1)).strftime('%Y-%m-%d')
                # Only apply the anomaly if the next day is within our dataframe
                if next_day in df['Date'].dt.strftime('%Y-%m-%d').values:
                    current_date_check = current_date.strftime('%Y-%m-%d')
                    # Calculate the new value as a percentage of the previous day's value
                    previous_value = df.loc[df['Date'] == current_date_check, 'Value'].values[0]
                    new_value = previous_value * (1 + anomaly_percentage/100)
                    new_value = round(float(new_value),2)  # Convert new_value to float
                    df['Value'] = df['Value'].astype(float)
                    df.loc[df['Date'] == next_day, 'Value'] = new_value
                current_date = datetime.strptime(next_day, '%Y-%m-%d')
        else:
            assert False, "Anomaly type not recognized"

        return df

    def prep_anomaly_data(self, df):
        def process_row(row):
            rows = []
            # Iterate over the first level of keys
            for main_key, nested_dict in row['anomaly_data'].items():
                # Iterate over the second level of keys, where each key is an anomaly dimension
                for anomaly_dimension, anomaly_dict in nested_dict.items():
                    new_row = row.copy()
                    new_row['anomaly'] = main_key
                    new_row['dimension'] = anomaly_dimension
                    # Iterate over the keys in the anomaly_dict and set them as new columns
                    for key, value in anomaly_dict.items():
                        new_row[key] = value
                    # Remove or transform the anomaly_data as per requirements here
                    del new_row['anomaly_data']
                    rows.append(new_row)
            return rows
        
        # Creating the new DataFrame
        new_rows = []
        for index, row in df.iterrows():
            new_rows.extend(process_row(row))
        
        # Convert list of dictionaries to DataFrame
        new_df = pd.DataFrame(new_rows)
        return new_df
    
    # Modify the check_row function to accept needle_number
    def check_row(self, row):
        if row['detected']:
            return 1
        else:
            return 10
        
    # Function to map score to color
    def score_to_color(self, score):
        if score <= 3.0:
            return 'Green'
        elif score <= 5.0:
            return 'Yellow-Green'
        elif score <= 7.0:
            return 'Yellow'
        elif score <= 9.0:
            return 'Orange'
        else:
            return 'Red'

    def generate_random_number(self, num_digits):
        lower_bound = 10**(num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            #raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
   

    def get_context_length_in_tokens(self, context):
        if (self.model_provider == "OpenAI") or ( self.model_provider == "Perplexity"):
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.enc.encode(context).ids)
        else:
            return len(self.enc.encode(context))
            #raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def get_tokens_from_context(self, context):
        if self.model_provider == "OpenAI":
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        elif self.model_provider == "GoogleVertex": 
            return self.enc.encode(context)
        else:

            return self.enc.encode(context)
            #raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    

    def generate_waffle_image(self, score_average_df, totals, run_name, df):
        # Assuming the DataFrame is named df and is already available
        # Totals list, e.g., totals = [100, 200, 300], one for each context_length
        # File path for saving the plot as a PNG file
        output_png_path = run_name + "_graph.png"  # Replace with your desired file path
        # Aggregate index positions and colors by context_length
        agg_df = score_average_df.groupby('context_length').agg({
            'index_position': lambda x: list(x),
            'color': lambda x: list(x)
        }).reset_index()

        # Initialize the config dictionary
        config = {}

        # Loop over each row in the aggregated DataFrame to populate the config
        for ind, row in agg_df.iterrows():
            context_length = row['context_length']
            index_positions = row['index_position']
            colors = row['color']
            
            # Initialize color boxes
            red_boxes = []
            green_boxes = []
            dark_yellow_boxes = []
            light_yellow_boxes = []
            orange_boxes = []
            
            # Populate color boxes based on the colors list
            for index, color in zip(index_positions, colors):
                if color == 'Red':
                    red_boxes.append(index)
                elif color == 'Green':
                    green_boxes.append(index)
                elif color == 'Yellow':
                    dark_yellow_boxes.append(index)
                elif color == 'Yellow-Green':
                    light_yellow_boxes.append(index)
                elif color == 'Orange':
                    orange_boxes.append(index)

            red_cnt = len(red_boxes)
            green_cnt = len(green_boxes)
            red_and_green_cnt = red_cnt + green_cnt
            if red_and_green_cnt == 0:
                frac_of_cnt = 1
            else:
                frac_of_cnt = green_cnt / red_and_green_cnt
            frac_of_cnt = int(frac_of_cnt*100)
            # Assuming the total for this context_length is provided externally
            total = totals[context_length]  # Adjust as needed based on how totals are provided
            #index = (len(agg_df) + 1 )*100 + 10 + (len(agg_df) - 1 - ind) + 1
            index = (len(agg_df) + 1, 1, (len(agg_df) - 1 - ind) + 1)
            false_positives = df[(df['context_length'] == context_length)].false_positive_slots.values[0]
            # Construct the config entry for this context_length
            #boxes_str = "green: " + str(len(green_boxes)) + " light yellow: " + str(len(light_yellow_boxes)) + " dark yellow: " + str(len(dark_yellow_boxes)) + " orange: " + str(len(orange_boxes)) + " red: " + str(len(red_boxes)) 
            config[index] = {
                'values': [total, 0, 0, 0],  # Example values, adjust as needed
                'labels': ["Detected Anomaly", "Missed Anomaly", "Normal Data", "False Positive"],
                'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
                'title': {'label': f'Context Length: {context_length} tokens with {total} timeseries in context window: {green_cnt} out of {red_and_green_cnt} : {frac_of_cnt}%', 'loc': 'left', 'fontsize': 22},
                'wspace': 4,
                'green_boxes': green_boxes,
                'light_yellow_boxes': light_yellow_boxes,
                'dark_yellow_boxes': dark_yellow_boxes,
                'orange_boxes': orange_boxes,
                'red_boxes': red_boxes,
                'false_positives': false_positives
            }
        
        #Render the waffle plot
        fig = plt.figure(
            FigureClass=Waffle,
            plots=config,
            rows=7,  # Outside parameter applied to all subplots, same as below
            cmap_name="Accent",  # Change color with cmap
            rounding_rule='ceil',  # Change rounding rule, so value less than 1000 will still have at least 1 block
            vertical=False,
            figsize=(6*4, 5*4)
        )
        #fig.suptitle('Timeseries Model Evals - Run: ' + run_name, fontsize=14, fontweight='bold',ha='left')
        # Save the figure to the specified file path
        plt.savefig(output_png_path, format='png')
        plt.show()        
  
        return True

    def get_results(self):
        return self.testing_results
        
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print(f" - Percentage Detect: {self.prompt_percentage_detect}, Anomaly Percentage: {self.anomaly_percentage} Noise Percent: {self.noise_level_percent} " )
        print ("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()

    RANDOM_NEEDLE_CITIES  = ['North Annmouth', 'Aarontown', 'East David', 'Jamesborough', 'Port Samanthaland', 'San Francisco', 'Andrewmouth', 'Port Whitneyland', 'Delhi', 
                        'Bangalore', 'Brianaberg', 'Richardfurt', 'Port Terri', 'Morrowfurt', 'East Patricia', 'Madrid', 'Michelleside', 'Tiffanyberg', 'North Emilyview',
                        'Yoderside', 'Jasonfort', 'Lake Michelle', 'Lisaborough', 'Suarezberg', 'Pittsview', 'Helsinki', 'East Judyfurt', 
                        'South Jamiemouth', 'Adamfurt', 'Port Vincent', 'Port Larry', 'Shawview', 'New Jenna', 'Port Samuel', 'Cohenton', 'Fosterchester', 'New Pamela', 
                        'Brownburgh', 'North Shannon', 'South Jessica', 'Robinsonburgh', 'Yangon', 'North Laura', 'Foxhaven', 'New Mark', 'Batesville', 'Istanbul', 
                        'Christophermouth', 'Lake Markbury', 'Romerochester', 'Millerside', 'Melbourne', 'South Amystad', 'Port Kelsey', 'West Jessica', 'Ho Chi Minh City',
                        'Port Benjaminburgh', 'Lake Austinland', 'Kimton', 'Terriborough', 'North Meredith', 'North Cesar', 'New Gregory', 'Port Austinside', 
                        'North Marilyn', 'Brookemouth', 'Nancyburgh', 'Brownstad', 'Doha', 'Port Amy', 'Edwardsshire', 'South Jason', 'Port Benjamin', 'Oliviamouth', 
                        'Martinstad', 'South Mark', 'Lake Michael', 'Russellhaven', 'New Adamtown', 'Owensstad', 'North Jonathan', 'West Amy', 'Browningport', 
                        'South Robinhaven', 'South Sean', 'West Bethany', 'South Richard', 'Terryburgh', 'Rothmouth', 'New Jennifer', 'North Tylerstad', 'Astana', 
                        'Christopherburgh', 'Catherineshire', 'West Connie', 'New Kristamouth', 'Karenshire', 'Lindatown', 'Nealberg', 'Moonburgh', 'Jenkinsside', 
                        'Mitchellbury', 'New Alexis', 'Victoria', 'North Rebeccaport', 'Robinsonhaven', 'Beirut', 'Williamville', 'Seoul', 'South Jeremyburgh', 'Mumbai',
                        'Robertsborough', 'West Carmenshire', 'South Sara', 'Bryanmouth', 'Janeland', 'South Andrewport', 'Lake Gina', 'Lake Russell', 'Port Abigail', 
                        'Port Kelly', 'Bangkok', 'West Catherineland', 'Dubai', 'Lake Edgarville', 'Veronicaborough', 'Amman', 'Christineside', 'West Kendrachester', 
                        'South Alisonburgh', 'Barbaraland', 'Whitakerton', 'Lake Miranda', 'Lake Julia', 'East Jamesmouth', 'Joyville', 'North David', 'Rossfurt', 
                        'South Debra', 'North Olivia', 'Michaelside', 'Moodyville', 'West John', 'North Matthewberg', 'North Bettyside', 'Whiteburgh', 'Kigali', 
                        'North Miranda', 'New James', 'North Audrey', 'West Joshuamouth', 'Lake Brittany', 'Johnsonmouth', 'Mcneilburgh', 'Thomasport', 'Lake Virginia', 
                        'East Kylemouth', 'North Kevinhaven', 'South Allison', 'Hamiltonfurt', 'Amsterdam', 'West Crystal', 'Khartoum', 'Nicholschester', 'Baghdad', 
                        'Buenos Aires', 'Rayport', 'East Susan', 'Bucharest', 'North Christina', 'Evanfurt', 'Michaelburgh', 'Justinmouth', 'East Scott', 'Mexico City',
                        'Antananarivo', 'Grayton', 'Seattle', 'Port Lisa', 'Elizabethfort', 'Ponceshire', 'Kampala', 'North Matthewview', 'West Robert',
                        'Jonathanfurt', 'Amberland', 'Jakarta', 'Woodstown', 'New Sheriburgh', 'Lake Darylmouth', 'Christopherbury', 'Jamesbury', 'South Timothy',
                        'Maryport', 'Lake Jenniferfort', 'South Nicholasburgh', 'Justinport', 'Greenville', 'East Justinton', 'Donaldberg', 'Jordanville', 'Cairo', 
                        'Laurashire', 'Ramirezfort', 'Sofia', 'Bowenview', 'Mossland', 'Chicago', 'Moscow', 'Scottland', 'Walterton', 'North Michaelmouth', 'Bowershaven', 
                        'Anthonyside', 'Diamondview', 'New Veronica', 'Port Christopher', 'North Felicia', 'North Joshua', 'West Marissachester', 'Ferrellfurt', 'Port Janet',
                        'North Chase', 'West Danielshire', 'Bairdhaven', 'Sherryborough', 'Colonchester', 'Lake Heatherbury', 'Hendersonbury', 'Port Josephhaven', 'Acostatown', 
                        'South Thomas', 'South Kathrynfurt', 'West Misty', 'Port Sophiachester', 'South Dylan', 'Athens', 'Kuala Lumpur', 'Sparksport', 'Middletonside', 'North April',
                        'New Shannonshire', 'Jeffreyborough', 'North Fernando', 'South Austinchester', 'South Staceyburgh', 'South Erictown', 'South Brittanyburgh', 'Walkerport', 
                        'South Brittney', 'Harrisberg', 'Matthewtown', 'Lake Dawnport', 'Goodmantown', 'Lake Jennifer', 'South Kevinville', 'Martinezmouth', 'New Jeffreyside', 'Thimphu', 
                        'South Cassidymouth', 'Dennisshire', 'Andrewberg', 'Lake Lisafort', 'Vienna', 'East Amber', 'Jonathonburgh', 'Lake Samantha', 'South Melanieport', 'North Lisa', 
                        'South Jesse', 'North Jenniferview', 'Dodsonchester', 'South Seanport', 'Shanechester', 'West Lisachester', 'Andersonfort', 'Lopezfort', 'West Carrie', 'Alexandermouth',
                        'Bonniefort', 'Kathyberg', 'Port Brian', 'Port Christine', 'East James', 'Copenhagen', 'Tokyo', 'East Teresamouth', 'Lake Jesse', 'Vancouver', 'Beckstad', 
                        'Brooksborough', 'Lake Shannonstad', 'Henryton', 'East Andrew', 'Fosterborough', 'Flynnfurt', 'Heatherborough', 'South Jamesmouth', 'Port Adam', 'Hollyberg', 
                        'Guzmanhaven', 'Lake Scottfurt', 'Port Jeffrey', 'Scottport', 'North Erin', 'East Ivan', 'Erikmouth', 'Michelleport', 'Aprilberg', 'West Jeremyville', 'Almaty',
                        'Lake Markberg', 'Port Daniel', 'New Erikborough', 'West Michaelton', 'South Gavinville', 'North Diane', 'North Michaelside', 'Charlesside', 'Timothymouth', 
                        'South Tyronechester', 'Bartlettville', 'New Alyssa', 'Ryanmouth', 'Nicolebury', 'Maputo', 'Lake Marisa', 'Vargasberg', 'East Yolandashire', 'Robertsonburgh',
                        'East Pamelamouth', 'Port Louis', 'Annemouth', 'Williamston', 'North Angelaberg', 'Belgrade', 'South Michael', 'Lake Destinyburgh', 'Mcclainland', 'Lake Trevorfurt',
                        'West George', 'Andersonmouth', 'East Christopher', 'Cooperland', 'North Laurahaven', 'Kathrynville', 'Lake Alexisside', 'Michaelbury', 'Diazfurt', 'South Joshualand',
                        'South Stephaniestad', 'Jenniferview', 'North Karen', 'Port Angela', 'New Davidside', 'South Darren', 'Port Sharon', 'New Jacob', 'Taylorbury', 
                        'East Jenniferchester', 'North Richard', 'West Dominiquehaven', 'Port Gary', 'East Nicholas', 'Port Madisonmouth', 'West Louis', 'South Robin', 'Pennyfort', 
                        'Berlin', 'Barcelona', 'Johannesburg', 'South Kristen', 'Port Joann', 'Nataliestad', 'North Meghanhaven', 'West Carlos', 'East Edward', 'Robertstad', 'Robertaport',
                        'Damascus', 'South Lanceton', 'Lynchburgh', 'Bradleyville', 'Brussels', 'Lake Margaret', 'East Cherylton', 'Colombo', 'Rabat', 'Lake Danielfort', 'Samanthaville', 
                        'Stevenberg', 'East Alisonton', 'Oliverland', 'Gomezmouth', 'Port Rickhaven', 'Santoschester', 'East Michael', 'Hendersontown', 'Budapest', 'Port Debbie', 'Adamburgh',
                        'New Jenniferfort', 'Richardberg', 'New Justinbury', 'Wellsborough', 'Kellishire', 'Wheelerfurt', 'Ethanland', 'Matthewside', 'Brookeside', 'Jenniferside', 
                        'East Shawn', 'West Jesus', 'South Tammieside', 'Justinshire', 'Ericaside', 'East Amandachester', 'East Michelleland', 'Kristenfort', 'West Josephside', 'Kathrynview',
                        'Wileyside', 'Stephaniechester', 'Christianmouth', 'Abigailport', 'Port Thomasbury', 'South Heatherhaven', 'Oslo', 'Lake Robertburgh', 'West Lindaberg', 'West Ashley', 
                        'Lake Melissa', 'South Larry', 'Sarajevo', 'Michaeltown', 'Montgomeryhaven', 'Cunninghamville', 'Briggsfurt', 'Port Michaeltown', 'Port Debra', 'Mendezbury', 
                        'Los Angeles', 'Timothyshire', 'New Jenniferstad', 'Gavinport', 'Port Carolyn', 'Lake Eric', 'Johnsonton', 'Sarahberg', 'Johnsonville', 'Yerevan', 'Santiago', 'Khanview',
                        'Longburgh', 'Margaretfurt', 'Mccormickport', 'Howardhaven', 'Bratislava', 'North Franciscoland', 'West Valerie', 'Meyerhaven', 'Jessicastad', 'New Laurie', 'Paris', 
                            'Danielstad', 'Port Laura', 'Sydney', 'Toronto', 'Duffyview', 'Shanghai', 'Tunis', 'Douglasmouth', 'Nelsonview', 'Lima', 'Lauriemouth', 'Shaffermouth', 'Nairobi', 'Kariville',
                            'Lisbon', 'Tashkent', 'Vargasstad', 'Andersonborough', 'Port John', 'Jennifershire', 'Hardinville', 'East Matthew', 'Lagos', 'Dakar', 'Chambersstad', 'Lisashire', 'Waltersborough', 
                            'Port Ruben', 'Matthewsburgh', 'North Michael', 'New Debra', 'West Charlesview', 'Port Garrett', 'Port Kenneth', 'South Williamville', 'Port Jonathanstad']

if __name__ == "__main__":
    #Runs Arize Phoenix Evaluation
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = LLMNeedleHaystackTester()

    ht.start_test()
    while(1):
        pass
