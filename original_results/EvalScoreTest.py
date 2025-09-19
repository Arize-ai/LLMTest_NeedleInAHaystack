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
from phoenix.experimental.evals.models import LiteLLMModel
from phoenix.experimental.evals.models.anthropic import AnthropicModel
from phoenix.experimental.evals.models.vertex import GeminiModel
import asyncio
import re
from phoenix.experimental.evals.utils import snap_to_rail
from phoenix.experimental.evals import (
    OpenAIModel,
    llm_generate,
)
import joypy
import matplotlib.cm as cm
from google.cloud import aiplatform
import vertexai.preview
import matplotlib.pyplot as plt
import random
import string

load_dotenv()

import os

class LLMNumericScoreEvalTester:
    """
    This class is used to test the LLM score Evals
    """

    def __init__(
        self,
        haystack_dir="PaulGrahamEssays",
        ###########################################
        ###### UNCOMMMENT only 1 Provider #########
        #model_provider = "OpenAI",
        #model_provider="Anthropic",
        # model_provider = "Perplexity",
        # model_provider = "Anyscale",
        #model_provider = "Mistral",
        # model_provider = "LiteLLM",
        model_provider = "GoogleVertex",
        #############################################
        ###### UNCOMMMENT only 1 model name #########
        # model_name='gpt-4',
        #model_name='gpt-4-1106-preview',
        #model_name='gpt-3.5-turbo-1106',
        #model_name="claude-2.1",
        model_name='gemini-pro',
        # model_name='gemini-pro-vision',
        # model_name='mistral/mistral-medium',
        #model_name='mistral/mistral-small',
        # model_name='mistral/mistral-tiny',
        # model_name='mistralai/Mistral-7B-Instruct-v0.1'
        # model_name='mistralai/Mixtral-8x7B-Instruct-v0.1'
        # model_name='together_ai/togethercomputer/llama-2-70b-chat',
        # model_name='huggingface/microsoft/phi-2',
        #############################################
        ## ERROR MODE Determines the type of test, we support 3
        #"spelling_errors" or "sadness or "frustration" 
        error_mode="frustration", 
        #range options: 1_to_10, 0_to_1, -1_to_1
        eval_score_range="-1_to_1",
        number_of_runs_per_context_length=5,
        #For spelling errors context token count can increase a lot, as error creates 2 tokens out of 1
        target_context_length=5000, 
        document_error_percent_min=0,
        document_error_percent_max=100,
        document_error_percent_intervals=10,  #How many intervals to run by % of corruption 
        # document_error_percent_intervals = 10, #Nice balance between speed and fidelity
        # document_error_percent_intervals = 35, #Uncomment for high fidelity run
        document_error_percents=None, #Not really used
         context_lengths=None, #Not really used
        document_error_percent_interval_type="linear",
        # google_project='', #Use OS env GOOGLE_PROJECT
        # google_location='', #Use OS env GOOGLE_LOCATION
        anthropic_template_version="simple",
        openai_api_key=None,
        anthropic_api_key=None,
        save_results=False,
        final_context_length_buffer=200,
        print_ongoing_status=True,
        results_version=1,
    ):
        """

        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param rnd_number_digits: The number of digits in the random number. Default is 7.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths: The lengths of the context. Default is None.
        :param document_error_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_error_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_error_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_error_percents: The depth percentages of the document. Default is None.
        :param document_error_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """

        self.document_error_percent_intervals = document_error_percent_intervals
        self.haystack_dir = haystack_dir
        self.results_version = results_version
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.anthropic_template_version = anthropic_template_version
        self.testing_results = []
        self.number_of_runs_per_context_length = number_of_runs_per_context_length
        self.error_mode = error_mode
        self.eval_score_range = eval_score_range

        self.target_context_length = target_context_length
        context_lengths_min = target_context_length
        context_lengths_max = target_context_length
        context_lengths_num_intervals= 1 #Fixing this to 1 for now
        self.context_lengths_num_intervals = 1 #Fixing this to 1 for now
        print("model_provider: " + model_provider)
        print("model_name: " + model_name)
        if context_lengths is None:
            if (
                context_lengths_min is None
                or context_lengths_max is None
                or context_lengths_num_intervals is None
            ):
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied."
                )
            else:
                self.context_lengths = np.round(
                    np.linspace(
                        context_lengths_min,
                        context_lengths_max,
                        num=context_lengths_num_intervals,
                        endpoint=True,
                    )
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_error_percents is None:
            if (
                document_error_percent_min is None
                or document_error_percent_max is None
                or document_error_percent_intervals is None
            ):
                raise ValueError(
                    "Either document_error_percent_min, document_error_percent_max, document_error_percent_intervals need to be filled out OR the document_error_percents needs to be supplied."
                )
            else:
                if document_error_percent_interval_type == "linear":
                    self.document_error_percents = np.round(
                        np.linspace(
                            document_error_percent_min,
                            document_error_percent_max,
                            num=document_error_percent_intervals,
                            endpoint=True,
                        )
                    ).astype(int)
                elif document_error_percent_interval_type == "sigmoid":
                    self.document_error_percents = [
                        self.logistic(x)
                        for x in np.linspace(
                            document_error_percent_min,
                            document_error_percent_max,
                            document_error_percent_intervals,
                        )
                    ]
        else:
            self.document_error_percents = document_error_percents

        if document_error_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_error_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_error_percent_intervals"
            )

        if model_provider not in [
            "OpenAI",
            "Anthropic",
            "Anyscale",
            "Perplexity",
            "GoogleVertex",
            "Mistral",
            "LiteLLM",
        ]:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

        if model_provider == "Anthropic" and "claude" not in model_name:
            raise ValueError(
                "If the model provider is 'Anthropic', the model name must include 'claude'. See https://docs.anthropic.com/claude/reference/selecting-a-model for more details on Anthropic models"
            )

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name

        if model_provider == "OpenAI":
            if not self.openai_api_key and not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "Either openai_api_key must be supplied with init, or OPENAI_API_KEY must be in env. Used for evaluation model"
                )
            else:
                self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        if self.model_provider == "Anthropic":
            if not self.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError(
                    "Either anthropic_api_key must be supplied with init, or ANTHROPIC_API_KEY must be in env."
                )
            else:
                self.anthropic_api_key = anthropic_api_key or os.getenv(
                    "ANTHROPIC_API_KEY"
                )

        if not self.model_name:
            raise ValueError("model_name must be provided.")

        if model_provider == "Anthropic":
            self.enc = Anthropic().get_tokenizer()
        elif model_provider == "OpenAI":
            self.enc = tiktoken.encoding_for_model(self.model_name)
        else:
            self.enc = tiktoken.encoding_for_model("gpt-4")

        self.google_project = os.getenv("GOOGLE_PROJECT")
        self.google_location = os.getenv("GOOGLE_LOCATION")

        if model_provider == "GoogleVertex":
            if not self.google_project:
                raise ValueError(
                    "Either google_project must be supplied with init, or GOOGLE_PROJECT must be in env."
                )
            if not self.google_location:
                raise ValueError(
                    "Either google_location must be supplied with init, or GOOGLE_LOCATION must be in env."
                )

        self.model_to_test_description = model_name

    def generate_random_number(self, num_digits):
        lower_bound = 10 ** (num_digits - 1)
        upper_bound = 10**num_digits - 1
        return random.randint(lower_bound, upper_bound)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    SIMPLE_TEMPLATE_SPELLING = """
            You are a helpful AI bot that checks for grammatical, spelling and typing errors in a document context. You are going to 
            score the document based on the percent of grammatical and typing errors. The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will be no grmatical errors in any word, a score of {templ_20_perc} will be 20% of words have errors,
            a {templ_50_perc} score will be 50% errors, a score of {templ_70_perc} is 70%, and a {templ_high} score will be all words in the context have a grammatical errors.
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return a score between {templ_high} and  {templ_low}, with the case of {templ_high} being all words have a grammatical error and {templ_low} being no words have a 
            grammatical or spelling errors.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: {templ_high}" or "the score is: {templ_low}"
            """

    SIMPLE_TEMPLATE_FRUSTRATION = """
            You are a helpful AI bot that detects frustrated conversations. You are going to 
            score the document based on the percent of sentences where the writer expresses frustration. 
            The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will indicate almost no frustrated sentences, a score of {templ_20_perc} will be 20% of sentences express frustration,
            a {templ_50_perc} score will be 50% of the sentnces express frustration, a score of {templ_70_perc} is 70%, and a {templ_high} score will be all the sentences, express frustration. 
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return a score between {templ_high} and {templ_low}, with the case of {templ_high} being all sentences express frustration and {templ_low} being no sentences express frustration.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: {templ_high}" or "the score is: {templ_low}"
    """

    SIMPLE_TEMPLATE_SADNESS = """
            You are a helpful AI bot that detects sadness and sorrow in writing. You are going to 
            score the document based on the percent of sentences where the writer expresses sadness or sorrow. 
            The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will indicate almost no sentences that have sadness or sorrow, a score of {templ_20_perc} will be 20% of sentences express sadness or sorrow,
            a {templ_50_perc} score will be 50% of the sentnces express sadness or sorrow, 
            a score of {templ_70_perc} is 70% have sadness or sorrow, and a {templ_high} score will be all the sentences, express sadness or sorrow. 
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return a score between {templ_high} and {templ_low}, with the case of {templ_high} being all sentences express sadness and sorrow and {templ_low} being no sentences express sadness or sorrow.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: {templ_high}" or "the score is: {templ_low}"
    """

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        contexts = []
        if self.error_mode == "spelling_errors":
            simple_template = self.SIMPLE_TEMPLATE_SPELLING
        elif self.error_mode == "frustration":
            simple_template = self.SIMPLE_TEMPLATE_FRUSTRATION
        elif self.error_mode == "sadness":
            simple_template = self.SIMPLE_TEMPLATE_SADNESS
        else:
            raise ValueError("template_version must be a valid template name")
        
        if self.eval_score_range == "1_to_10":
            templ_range = 10
            #10 is 100%, 7 is 70%, 5 is 50%, 2  is 20%, 1 is 0%
            simple_template = simple_template.format(templ_high=str(10), templ_low=str(1), templ_20_perc=str(2), 
                                                                    templ_50_perc=str(5), templ_70_perc=str(7),
                                                                    context="{context}")
        elif self.eval_score_range == "0_to_1":
            templ_range = 1
            #1 is 100%, 0.7 is 70%, 0.5 is 50%, 0.2  is 20%, 0.1 is 0%
            simple_template = simple_template.format(templ_high=str(1), templ_low=str(0), templ_20_perc=str(0.2), 
                                                                    templ_50_perc=str(0.5), templ_70_perc=str(0.7),
                                                                    context="{context}")     
        elif self.eval_score_range == "-1_to_1":
            templ_range = 2
            #1 is 100%, 0.4 is 70%, 0 is 50%, -0.4  is 20%, -1 is 0%
            simple_template = simple_template.format(templ_high=str(1), templ_low=str(-1), templ_20_perc=str(-0.4), 
                                                                    templ_50_perc=str(0), templ_70_perc=str(0.4),
                                                                    context="{context}")
        # Evaluation of the model performance
        # Uses Phoenix Evals
        if self.model_provider == "OpenAI":
            model = OpenAIModel(model_name="gpt-4-1106-preview")
            template = simple_template
        elif self.model_provider == "Anthropic":
            model = AnthropicModel(model="claude-2.1", temperature=0.0)
            # model = LiteLLMModel(model_name="claude-2.1", temperature=0.0)
            template = simple_template
 
        elif self.model_provider == "LiteLLM":
            model = LiteLLMModel(model_name=self.model_name, temperature=0.0)
            template = simple_template
            litellm.set_verbose = True
            litellm.vertex_project = self.google_project
            litellm.vertex_location = self.google_location

        elif self.model_provider == "GoogleVertex":
            template = simple_template
            aiplatform.init(
                # your Google Cloud Project ID or number
                # environment default used is not set
                project=self.google_project,
                # the Vertex AI region you will use
                # defaults to us-central1
                location=self.google_location,
            )
            model = GeminiModel()
        else:
            model = LiteLLMModel(model_name=self.model_name, temperature=0.0)
            # litellm.set_verbose=True
            template = simple_template

        full_context = self.read_context_files()
        for context_length in self.context_lengths:
            for run_number in range(self.number_of_runs_per_context_length):
                trim_context = self.encode_and_trim(full_context, context_length)
                for error_percent in self.document_error_percents:
                    print("context length: " + str(context_length))
                    print("error_percent : " + str(error_percent))
                    print("run_number : " + str(run_number))
                    results = self.create_contexts(
                        trim_context, context_length, error_percent, self.error_mode, run_number
                    )
                    contexts.append(results)
        df = pd.DataFrame(contexts)
        # The rails is used to search outputs for specific values and return a binary value
        # It will remove text such as ",,," or "..." and general strings from outputs
        # It answers needle_rnd_number or unanswerable or unparsable (if both or none exist in output)

        def numeric_score_eval(output, row_index):
            # This is the function that will be called for each row of the dataframe
            row = df.iloc[row_index]
            # needle = row['needle_rnd_number']
            # The rails is used to search outputs for specific values and returns needle, unsanswerable, or unparsable
            # railed_output = snap_to_rail(output, [needle, "UNANSWERABLE"])
            print("The error percent is: " + str(row["corruption_percentage"]))
            print(f"🔍 The model output is: {output}")
            score = self.find_score(output)
            print(f"🔍 The score is: {score}")
            # If the needle is in the output, then it is answerable

            # If the needle is not in the output, then it is unanswerable
            print(
                "---------------------------------------------------------------------"
            )
            print(f"Row details: ")
            print(row)
            return {"score": score}

        # This is the core of the Phoenix evaluation
        # It runs the model on every row of the dataframe
        # It looks for columns that are defined in the template question/context
        # The generation of the model, the output, is "cleaned" up by the rails
        # The rails are used to search for specific values in the output
        # The output is then classified as either needle_rnd_number, unanswerable, or unparsable
        # This runs a number of threads in parallel speeding up the generation/Evaluation process
        #nest_asyncio.apply()  # Run async
        test_results = llm_generate(
            dataframe=df,
            template=template,
            model=model,
            verbose=True,
            concurrency=15,
            # Callback function that will be called for each row of the dataframe
            output_parser=numeric_score_eval,
            # These two flags will add the prompt / response to the returned dataframe
            include_prompt=True,
            include_response=True,
        )
        run_name = (
            "templ_"
            + self.error_mode
            + "_"
            + self.eval_score_range
            + "_"
            + self.model_provider
            + "_"
            + self.model_name
            + "_"
            + str(self.context_lengths_num_intervals)
            + "_"
            + str(self.document_error_percent_intervals)
        ).replace("/", "_")
        df = pd.concat([df, test_results], axis=1)
        jitter_magnitude = 0.04*(templ_range)/10  #We had a bug that made -1 to 1 look worse
        df['score_jitter'] = df['score'] + np.random.uniform(
            -jitter_magnitude, jitter_magnitude, size=len(df))       
        df['dp_string'] = df['corruption_percentage'].astype(str) 
        self.plot_point_distribution(
            df,
            "score_jitter",
            "corruption_percentage",
            run_name,
            self.error_mode,
            circle_size=250,
            swap_axes=True,
            show_only_medians=True,
            error_bars=False,
            model_name=self.model_name,
            eval_score_range = self.eval_score_range
        )

        df_sorted = df.sort_values(by='corruption_percentage', ascending=False)
        fig, axes = joypy.joyplot(df_sorted,by="dp_string", column="score_jitter", overlap=0.1,  
                      colormap=cm.Blues_r, linecolor="grey", linewidth=1,ylim="own")
        
        # Save the figure
        plt.savefig(run_name + '_joyplot.png', dpi=300)  # You can change the file name and DPI as needed

        # Show the plot
        #plt.show()
        df.to_csv("save_results_" + run_name + "_.csv")

        return contexts

    # Modify the check_row function to accept needle_number
    def check_row(self, row):
        if row["insert_needle"]:
            # needle is inserted so check for the needle
            if row["label"] == row["needle_rnd_number"]:
                return 1
            elif row["label"] == "unanswerable":
                return 10
            elif row["label"] == "NOT_PARSABLE":
                return 5
            else:
                return 5
        else:
            # needle is not inserted so check for unanswerable
            return 1 if row["label"] == "unanswerable" else 10

    def create_contexts(self, trim_context, context_length, error_percent, error_mode, run_number):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, error_percent):
                return
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(trim_context, error_percent, error_mode)
        results = {
            "context": context,  # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            "model": self.model_to_test_description,
            "context_length_limit": int(context_length),
            "context_length": len(self.get_tokens_from_context(context)),
            "corruption_percentage": str(error_percent),
            "version": self.results_version,
            "run_number": run_number,
        }
        return results

    def result_exists(self, context_length, error_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = "results/"
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    context_length_met = result["context_length"] == context_length
                    error_percent_met = result["error_percent"] == error_percent
                    version_met = result.get("version", 1) == self.results_version
                    model_met = result["model"] == self.model_name
                    if (
                        context_length_met
                        and error_percent_met
                        and version_met
                        and model_met
                    ):
                        return True
        return False

    def generate_context(self, trim_context, error_percent, error_mode):
        if error_mode == "spelling_errors":
            # Insert your random statement according to your depth percent
            context = self.insert_errors_in_paragraph(trim_context, error_percent)
        else:
            context = self.insert_sentiment(trim_context, error_percent, error_mode)

        return context

    def encode_text_to_tokens(self, text):
        if self.model_provider == "OpenAI":
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            return self.enc.encode(text)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def insert_errors_in_paragraph(self, paragraph, percent_error):
        """Inserts grammatical errors into a given percentage of words in a paragraph."""
        words = paragraph.split()
        num_words = len(words)
        num_errors = int(num_words * percent_error / 100)

        # Select random indices for the words to which we will apply errors
        error_indices = random.sample(range(num_words), num_errors)
        for i in error_indices:
            word_to_insert = self.insert_error_in_word(words[i])
            words[i] = word_to_insert
        context_to_return = " ".join(words)
        return context_to_return

    def insert_error_in_word(self, word):
        """Inserts a grammatical error into a given word, with an additional error type to double a letter."""
        # Randomly choose the type of error to introduce
        error_type = random.choice(["remove", "add", "swap", "double"])

        if error_type == "remove":
            # Remove a random letter from the word (if it's not a single letter)
            if len(word) > 1:
                remove_index = random.randint(0, len(word) - 1)
                return word[:remove_index] + word[remove_index + 1 :]
            else:
                # Cannot remove from a single letter, choose another error
                error_type = "add"

        if error_type == "add":
            # Add a random letter at a random position in the word
            add_index = random.randint(0, len(word))
            random_letter = random.choice(string.ascii_letters)
            return word[:add_index] + random_letter + word[add_index:]

        if error_type == "swap":
            # Swap two adjacent letters in the word (if it has at least two letters)
            if len(word) > 1:
                swap_index = random.randint(0, len(word) - 2)
                return (
                    word[:swap_index]
                    + word[swap_index + 1]
                    + word[swap_index]
                    + word[swap_index + 2 :]
                )
            else:
                # Cannot swap in a single letter, choose another error
                return self.insert_error_in_word(word)  # Recurse with the same word

        if error_type == "double":
            # Double a random letter in the word
            double_index = random.randint(0, len(word) - 1)
            return (
                word[:double_index] + word[double_index] * 2 + word[double_index + 1 :]
            )

        return word

    def find_score(self, output):
        # Regular expression pattern
        # It looks for 'score is', followed by any characters (.*?), and then a float or integer
        pattern = r"score is.*?([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"

        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            # Extract and return the number
            return float(match.group(1))
        else:
            return None

    def insert_sentiment(self, paragraph, frustration_percent, error_mode):
        # List of frustration expressions
        if error_mode == "frustration":
            expression_list = self.FRUSTRATION_EXPRESSIONS
        elif error_mode == "sadness":
            expression_list = self.SADNESS_AND_SORROW_EXPRESSIONS
        else:
            raise ValueError("error_mode must be either 'frustration' or 'confusion'")

        # Function to split the paragraph into sentences
        def split_into_sentences(text):
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
            return sentences

        # Split the paragraph into sentences
        sentences = split_into_sentences(paragraph)
        modified_sentences = []

        for sentence in sentences:
            if random.uniform(0, 100) <= frustration_percent:
                expression = random.choice(expression_list)
                add_at_end = random.choice([True, False])

                if sentence.endswith('.'):
                    sentence = sentence[:-1]  # Remove the period

                if add_at_end:
                    # Append sentiment expression at the end
                    sentence = f"{sentence}, {expression}."
                else:
                    # Append sentiment expression at the beginning
                    sentence = f"{expression}, {sentence}."

            modified_sentences.append(sentence)

        # Combine the modified sentences back into a paragraph
        modified_paragraph = ' '.join(modified_sentences)
        return modified_paragraph


    def get_context_length_in_tokens(self, context):
        if (self.model_provider == "OpenAI") or (self.model_provider == "Perplexity"):
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return len(self.enc.encode(context).ids)
        else:
            return len(self.enc.encode(context))
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider == "OpenAI":
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            return self.enc.encode(context)
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider == "OpenAI":
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            return self.enc.decode(tokens[:context_length])
            # raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context


    def calculate_errors(self, dataframe, group_column, value_column):
        grouped = dataframe.groupby(group_column)
        group_sizes = grouped.size()
        medians = grouped[value_column].median()
        lower_quartiles = grouped[value_column].quantile(0.25)
        upper_quartiles = grouped[value_column].quantile(0.75)

        #print("Group sizes:\n", group_sizes)
        #print("Lower quartiles:\n", lower_quartiles)
        #print("Medians:\n", medians)
        #print("Upper quartiles:\n", upper_quartiles)

        lower_errors = medians - lower_quartiles
        upper_errors = upper_quartiles - medians

        # Ensuring that no error value is less than 0.1
        lower_errors = np.maximum(lower_errors, 0.2)
        upper_errors = np.maximum(upper_errors, 0.2)

        return lower_errors , upper_errors,

    def plot_point_distribution(
            self,
            dataframe,
            x_column,
            y_column,
            run_name,
            error_mode,
            circle_size=225,
            swap_axes=False,
            show_only_medians=False,
            error_bars=False,
            model_name="",
            eval_score_range = "1_to_10",
        ):
            dataframe[y_column] = pd.to_numeric(dataframe[y_column], errors="coerce")
            clean_df = dataframe.dropna(subset=[x_column, y_column])
            clean_df = clean_df.sort_values(by=y_column, ascending=True)
            df_median = clean_df[[x_column, y_column]].groupby(y_column).median()

            x_min = clean_df[x_column].min() - 1
            x_max = clean_df[x_column].max() + 1
            if eval_score_range == "1_to_10":
                x_max = np.maximum(x_max, 10)
            elif eval_score_range == "0_to_1":
                x_max = np.maximum(x_max, 1)
            elif eval_score_range == "-1_to_1":
                x_max = np.maximum(x_max, 1)
                
            

            lower_errors, upper_errors = self.calculate_errors(clean_df, y_column, x_column)

            fig, ax = plt.subplots(figsize=(16, 10), dpi=80)

            for i, (idx, row) in enumerate(df_median.iterrows()):
                if error_bars:
                        if swap_axes:
                            ax.errorbar(
                                y=df_median.loc[idx, x_column], x=i,
                                yerr=[[lower_errors[idx]], [upper_errors[idx]]],  # Vertical error
                                fmt='o', color='firebrick',
                                capsize=5  # Set a visible cap size
                            )
                        else:
                            ax.errorbar(
                                x=df_median.loc[idx, x_column], y=i,
                                xerr=[[lower_errors[idx]], [upper_errors[idx]]],  # Horizontal error
                                fmt='o', color='firebrick',
                                capsize=5  # Set a visible cap size
                        )
                if not show_only_medians:
                    df_category = clean_df[clean_df[y_column] == idx]

                    if swap_axes:
                        ax.scatter(
                            y=df_category[x_column],  # Use original x_column values
                            x=np.repeat(i, df_category.shape[0]),
                            s=circle_size,
                            edgecolors="gray",
                            color=(1, 0.5, 0, 0.5),
                            alpha=0.3,
                        )
                    else:
                        ax.scatter(
                            x=df_category[x_column],  # Use original x_column values
                            y=np.repeat(i, df_category.shape[0]),
                            s=circle_size,
                            edgecolors="gray",
                            color=(1, 0.5, 0, 0.5),
                            alpha=0.3,
                        )


                if swap_axes:
                    ax.scatter(
                        y=df_median.loc[idx, x_column], x=i, s=circle_size, c="grey"
                    )
                else:
                    ax.scatter(
                        x=df_median.loc[idx, x_column], y=i, s=circle_size, c="grey"
                    )

            if swap_axes:
                ax.set_ylabel(f"LLM Eval {x_column} of {error_mode}", alpha=0.7)
                ax.set_xlabel(f"LLM Eval {y_column} of {error_mode}", alpha=0.7)
                ax.set_ylim(x_min, x_max)
                ax.set_xlim(-1, len(df_median))
                ax.set_xticks(range(len(df_median)))
                ax.set_xticklabels(
                    [f"{idx:.1f}" for idx in df_median.index],
                    fontdict={"horizontalalignment": "right"},
                    alpha=0.7,
                )
            else:
                ax.set_xlabel(f"LLM Eval {x_column} of {error_mode} from model {model_name}", alpha=0.7)
                ax.set_xlim(x_min, x_max)
                ax.set_ylabel(f"LLM Eval {y_column}", alpha=0.7)
                ax.set_yticks(range(len(df_median)))
                ax.set_yticklabels(
                    [f"{idx:.1f}" for idx in df_median.index],
                    fontdict={"horizontalalignment": "right"},
                    alpha=0.7,
                )

            red_patch = plt.plot(
                [],
                [],
                marker="o",
                ms=10,
                ls="",
                mec=None,
                color="grey",
                label="Median",
            )
            if not show_only_medians:
                orange_patch = plt.plot(
                    [],
                    [],
                    marker="o",
                    ms=10,
                    ls="",
                    mec=None,
                    color=(1, 0.5, 0, 0.5),
                    label="Data Points",
                )
                plt.legend(handles=[red_patch[0], orange_patch[0]], loc="lower right")
            else:
                plt.legend(handles=[red_patch[0]], loc="lower right")

            #ax.set_title(f"Distribution of Eval {x_column} by {y_column} insertion of " + error_mode + " data", fontdict={"size": 22})

            plt.xticks(alpha=0.7)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["bottom"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.grid(axis="both", alpha=0.4, linewidth=0.1)

            # File path for saving the plot as a PNG file
            output_png_path = run_name + "_graph.png"
            plt.savefig(output_png_path, bbox_inches="tight")

            #plt.show()




    FRUSTRATION_EXPRESSIONS = [
        "Ugh",
        "Why did you do that",
        "Who do you think you are",
        "That is lame",
        "Seriously",
        "This is ridiculous",
        "Can you not",
        "What the heck",
        "I can't even",
        "Oh, come on",
        "Not this again",
        "This is driving me crazy",
        "Are you kidding me",
        "I've had it",
        "This is absurd",
        "What a pain",
        "I'm so done with this",
        "Why me",
        "This is the last thing I need",
        "This is nonsense",
        "I'm losing my patience",
        "Not now",
        "Great, just great",
        "I don’t believe this",
        "This is so annoying",
        "Can't anything go right",
        "This is hopeless",
        "I'm fed up",
        "Enough already",
        "This is the worst",
        "I can't believe this",
        "What a mess",
        "This is a disaster",
        "Not again",
        "This is unbearable",
        "Why is this happening",
        "I'm so over this",
        "This is a joke, right",
        "I'm at my wit's end",
        "You've got to be kidding me",
        "This makes no sense",
        "I'm beyond frustrated",
        "This is infuriating",
        "I've had enough",
        "What a nightmare",
        "This is unacceptable",
        "I can't take this anymore",
        "This is a total headache",
        "Why is everything so complicated",
        "I'm so irritated right now"
    ]


    SADNESS_AND_SORROW_EXPRESSIONS = [
        "I'm heartbroken",
        "This is so sad",
        "I feel so down",
        "I'm in tears",
        "This is heartbreaking",
        "I'm grieving",
        "I feel so empty",
        "This is so disheartening",
        "I'm in despair",
        "I'm feeling blue",
        "This is so depressing",
        "I'm mourning",
        "I'm so melancholy",
        "This is so tragic",
        "I'm in sorrow",
        "I feel so low",
        "This is so gloomy",
        "I'm weeping",
        "I feel so desolate",
        "This is so mournful",
        "I'm in anguish",
        "I feel so hopeless",
        "This is so dismal",
        "I'm in distress",
        "I feel so forlorn",
        "This is so sorrowful",
        "I'm in pain",
        "I feel so dejected",
        "This is so lugubrious",
        "I'm in agony",
        "I feel so bereaved",
        "This is so doleful",
        "I'm suffering",
        "I feel so crushed",
        "This is so bleak",
        "I'm in torment",
        "I feel so woeful",
        "This is so heavy-hearted",
        "I'm in a funk",
        "I feel so somber",
        "This is so dolorous",
        "I'm lamenting",
        "I feel so wretched",
        "This is so cheerless",
        "I'm despondent",
        "I feel so pained",
        "This is so crestfallen",
        "I'm in a state of despair",
        "I feel so gutted",
        "This is so melancholic"
    ]




    def print_start_test_summary(self):
        print("\n")
        print("Starting In a test for score based Evals ...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_error_percents)}, Min: {min(self.document_error_percents)}%, Max: {max(self.document_error_percents)}%"
        )
        print("Error Mode " + self.error_mode)
        print("Eval Score Range " + self.eval_score_range)
        print("Model Provider " + self.model_provider)
        print("Number of statistical runs per range " + str(self.number_of_runs_per_context_length))
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()


if __name__ == "__main__":
    # Runs Arize Phoenix Evaluation
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    ht = LLMNumericScoreEvalTester()

    ht.start_test()
