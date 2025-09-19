import glob
import os
import random
import re
import nest_asyncio
from phoenix.evals import (
    OpenAIModel,
    AnthropicModel,
    llm_generate,
)

import tiktoken
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.anthropic import AnthropicInstrumentor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

class LLMNumericScoreEvalTester:
    def __init__(
        self,
        haystack_dir="PaulGrahamEssays",

        # model_provider = "Qwen",
        # model_name = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        # model_name = "fireworks/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        
        # model_provider = "Anthropic",
        # model_name = "claude-opus-4-20250514",
        # model_name = "claude-sonnet-4-20250514",

        model_provider = "OpenAI",
        model_name='gpt-4',
        # model_name='o3',
        # model_name='gpt-5-nano',
        
        error_mode="frustration", 
        eval_score_range="1_to_10",
        number_of_runs_per_context_length=2,  
        target_context_length=5000,
        document_error_percent_min=0,
        document_error_percent_max=100,
        document_error_percent_intervals=10,
        openai_api_key=None,
        anthropic_api_key=None,
        print_ongoing_status=True,
    ):

        self.document_error_percent_intervals = document_error_percent_intervals
        self.haystack_dir = "PaulGrahamEssays"
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.number_of_runs_per_context_length = number_of_runs_per_context_length
        self.error_mode = error_mode
        self.eval_score_range = eval_score_range
        self.target_context_length = target_context_length
        self.context_lengths = [target_context_length]
        
        print(f"model_provider: {model_provider}")
        print(f"model_name: {model_name}")
        
        self.document_error_percents = np.round(
            np.linspace(
                document_error_percent_min,
                document_error_percent_max,
                num=document_error_percent_intervals,
                endpoint=True,
            )
        ).astype(int)

        if model_provider not in ["OpenAI", "Anthropic", "Qwen"]:
            raise ValueError("model_provider must be one of: OpenAI, Anthropic, Qwen")

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model_name = model_name

        if model_provider == "OpenAI" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be provided or set in environment")
        if model_provider == "Anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY must be provided or set in environment")

        if model_provider == "OpenAI":
            self.enc = tiktoken.encoding_for_model("gpt-4")
            self.max_model_tokens = 128_000
        elif model_provider == "Anthropic":
            self.enc = tiktoken.encoding_for_model("gpt-4")
            self.max_model_tokens = 200_000
        elif model_provider == "Qwen":
            self.enc = tiktoken.encoding_for_model("gpt-4")
            self.max_model_tokens = 32_768
        else:
            self.enc = tiktoken.encoding_for_model("gpt-4")

        self.model_to_test_description = model_name


    SIMPLE_TEMPLATE_SPELLING = """
            You are a helpful AI bot that checks for grammatical, spelling and typing errors in a document context. You are going to 
            score the document based on the percent of words that contain grammatical and typing errors. 
            The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will be no words with grammatical errors, a score of {templ_20_perc} will be 20 percent of words have errors,
            a {templ_50_perc} score will be 50 percent of words have errors, a score of {templ_70_perc} is 70 percent of words have errors, and a {templ_high} score will be all the words in the context have grammatical errors.
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return any score between {templ_high} and  {templ_low}, with the case of {templ_high} being all words have grammatical and spelling errors and {templ_low} being no words have grammatical or spelling errors.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: [SCORE]"
            """

    SIMPLE_TEMPLATE_FRUSTRATION = """
            You are a helpful AI bot that detects frustrated conversations. You are going to 
            score the document based on the percent of sentences where the writer expresses frustration. 
            The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will indicate almost no frustrated sentences, a score of {templ_20_perc} will be 20 percent of sentences express frustration,
            a {templ_50_perc} score will be 50 percent of the sentences express frustration, a score of {templ_70_perc} is 70 percent, and a {templ_high} score will be all the sentences, express frustration. 
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return any score between {templ_high} and {templ_low}, with the case of {templ_high} being all sentences express frustration and {templ_low} being no sentences express frustration.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: {templ_high}" or "the score is: {templ_low}"
    """

    SIMPLE_TEMPLATE_SADNESS = """
            You are a helpful AI bot that detects sadness and sorrow in writing. You are going to 
            score the document based on the percent of sentences where the writer expresses sadness or sorrow. 
            The score should be between {templ_high} and {templ_low}.
            A {templ_low} score will indicate almost no sentences that have sadness or sorrow, a score of {templ_20_perc} will be 20 percent of sentences express sadness or sorrow,
            a {templ_50_perc} score will be 50 percent of the sentences express sadness or sorrow, a score of {templ_70_perc} is 70 percent have sadness or sorrow, 
            and a {templ_high} score will be all the sentences, express sadness or sorrow. 
            The following is the document context.
        
            #CONTEXT
            {context}
            #ENDCONTEXT

            #QUESTION
            Please return a score between {templ_high} and {templ_low}, with the case of {templ_high} being all sentences express sadness and sorrow and {templ_low} being no sentences express sadness or sorrow.
            You will return no other text or language besides the score. Only return the score. 
            Please return in a format that is "the score is: [SCORE]"
    """

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        print("Running test...")
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
            #More intuitive linear scale: 1=0%, 3=20%, 6=50%, 8=70%, 10=100%
            #But with clearer mapping instructions
            simple_template = simple_template.format(templ_high=str(10), templ_low=str(1), templ_20_perc=str(2.8), 
                                                                    templ_50_perc=str(5.5), templ_70_perc=str(7.3),
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
        elif self.eval_score_range == "A_to_E":
            templ_range = 5
            #A is 100% (worst - completely error-filled), B is 80%, C is 60%, D is 40%, E is 20% (best - mostly clean)
            #Mapping: A=100%, B=80%, C=60%, D=40%, E=20%
            simple_template = simple_template.format(templ_high="A", templ_low="E", templ_20_perc="E", 
                                                                    templ_50_perc="C", templ_70_perc="B",
                                                                    context="{context}")
        print("simple_template: " + simple_template)
        # Evaluation of the model performance using Phoenix Evals
        if self.model_provider == "OpenAI":
            model = OpenAIModel(model=self.model_name, temperature=1.0)
            template = simple_template
        elif self.model_provider == "Anthropic":
            model = AnthropicModel(model=self.model_name)
            template = simple_template
        elif self.model_provider == "Qwen":
            if self.model_name.startswith("fireworks/"):
                print(f"Using Qwen via Fireworks API: {self.model_name}")
                model = OpenAIModel(
                    model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
                    temperature=0.6,
                    base_url="https://api.fireworks.ai/inference/v1",
                    api_key=os.getenv("FIREWORKS_API_KEY")
                )
                template = simple_template
            else:
                together_model = self.model_name.split("together_ai/")[-1]
                print(f"Using Qwen via Together OpenAI-compatible API with model: {together_model}")
                model = OpenAIModel(
                    model=together_model,
                    temperature=0.0,
                    base_url="https://api.together.xyz/v1",
                    api_key=os.getenv("TOGETHER_API_KEY"),
                )
                template = simple_template

        full_context = self.read_context_files()
        for context_length in self.context_lengths:
            print("context_length: " + str(context_length))
            for run_number in range(self.number_of_runs_per_context_length):
                trim_context = self.encode_and_trim(full_context, context_length)
                for error_percent in self.document_error_percents:
                    results = self.create_contexts(
                        trim_context, context_length, error_percent, self.error_mode, run_number
                    )
                    contexts.append(results)
        df = pd.DataFrame(contexts)
    
        def numeric_score_eval(output, row_index):
            row = df.iloc[row_index]
            print("The error percent is: " + str(row["corruption_percentage"]))
            print(f"🔍 The model output is: {output}")
            score = self.find_score(output)
            print(f"🔍 The score is: {score}")
            print(
                "---------------------------------------------------------------------"
            )
            print("Row details: ")
            print(row)
            return {"score": score}

        # Run the model on every row of the dataframe
        nest_asyncio.apply()
        test_results = llm_generate(
            dataframe=df,
            template=template,
            model=model,
            verbose=False,
            concurrency=1,
            output_parser=numeric_score_eval,
            include_prompt=True,
            include_response=True,
        )
        run_name = (
            self.model_name
            + "_"
            + self.error_mode
            + "_"
            + self.eval_score_range
        ).replace("/", "_").replace("-", "_")
        df = pd.concat([df, test_results], axis=1)
        
        # Check if we have any successful results
        if 'score' not in df.columns or df['score'].isna().all():
            print("WARNING: No successful model responses. Check your API key and model configuration.")
            print("Available columns:", df.columns.tolist())
            return contexts
        
        # Handle jitter differently for letter grades vs numeric scores
        if self.eval_score_range == "A_to_E":
            # For letter grades, we'll convert to numeric for jitter, then back to letters
            # Mapping: A=100%, B=80%, C=60%, D=40%, E=20%
            letter_to_num = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}
            num_to_letter = {100: 'A', 80: 'B', 60: 'C', 40: 'D', 20: 'E'}
            
            # Convert scores to numeric for jitter calculation
            df['score_numeric'] = df['score'].map(letter_to_num)
            jitter_magnitude = 15
            df['score_jitter_numeric'] = df['score_numeric'] + np.random.uniform(
                -jitter_magnitude, jitter_magnitude, size=len(df))
            
            # Convert back to letters (round to nearest 20, then map back)
            df['score_jitter_numeric'] = (df['score_jitter_numeric'] / 20).round() * 20
            df['score_jitter_numeric'] = df['score_jitter_numeric'].clip(20, 100)  # Ensure within range
            df['score_jitter'] = df['score_jitter_numeric'].map(num_to_letter)
            df = df.drop(['score_numeric', 'score_jitter_numeric'], axis=1)
        else:
            # For numeric scores, use the original jitter calculation
            jitter_magnitude = 0.04*(templ_range)/10
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
        
        # Create violin plot to show distribution shapes
        self.plot_violin_distribution(
            df,
            "score_jitter",
            "corruption_percentage",
            run_name,
            self.error_mode,
            model_name=self.model_name,
            eval_score_range=self.eval_score_range
        )

        df.to_csv("save_results_" + run_name + "_.csv")
        return contexts


    def create_contexts(self, trim_context, context_length, error_percent, error_mode, run_number):
        context = self.generate_context(trim_context, error_percent, error_mode)
        results = {
            "context": context,
            "model": self.model_to_test_description,
            "context_length_limit": int(context_length),
            "context_length": len(self.get_tokens_from_context(context)),
            "corruption_percentage": str(error_percent),
            "run_number": run_number,
        }
        return results


    def generate_context(self, trim_context, error_percent, error_mode):
        if error_mode == "spelling_errors":
            # Insert your random statement according to your depth percent
            context = self.insert_errors_in_paragraph(trim_context, error_percent)
        else:
            context = self.insert_sentiment(trim_context, error_percent, error_mode)

        return context

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
            random_letter = random.choice("abcdefghijklmnopqrstuvwxyz")
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
        # Handle different score formats based on eval_score_range
        if self.eval_score_range == "A_to_E":
            # Look for letter grades A, B, C, D, E
            pattern = r"score is.*?([A-E])"
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                score = match.group(1).upper()
                print(f"Extracted letter score: {score} from output: {output[:100]}...")
                return score
            else:
                print(f"WARNING: Could not extract letter score from output: {output[:100]}...")
                return None
        else:
            # Regular expression pattern for numeric scores
            # It looks for 'score is', followed by any characters (.*?), and then a float or integer
            pattern = r"score is.*?([+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"

            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                # Extract and return the number
                score = float(match.group(1))
                print(f"Extracted score: {score} from output: {output[:100]}...")
                return score
            else:
                print(f"WARNING: Could not extract score from output: {output[:100]}...")
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
        return len(self.enc.encode(context))

    def read_context_files(self):
        print("reading context files!")
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context)

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length])

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context


    def calculate_errors(self, dataframe, group_column, value_column):
        grouped = dataframe.groupby(group_column)
        medians = grouped[value_column].median()
        lower_quartiles = grouped[value_column].quantile(0.25)
        upper_quartiles = grouped[value_column].quantile(0.75)

        lower_errors = medians - lower_quartiles
        upper_errors = upper_quartiles - medians

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
            # Handle letter grades differently
            if eval_score_range == "A_to_E":
                # For letter grades, we need to convert to numeric for plotting
                # Mapping: A=100%, B=80%, C=60%, D=40%, E=20%
                letter_to_num = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}
                dataframe = dataframe.copy()
                dataframe[x_column + '_numeric'] = dataframe[x_column].map(letter_to_num)
                x_column_numeric = x_column + '_numeric'
                
                # Use numeric column for calculations
                dataframe[y_column] = pd.to_numeric(dataframe[y_column], errors="coerce")
                clean_df = dataframe.dropna(subset=[x_column_numeric, y_column])
                clean_df = clean_df.sort_values(by=y_column, ascending=True)
                df_median = clean_df[[x_column_numeric, y_column]].groupby(y_column).median()
                
                # For A-E scoring, we'll set proper limits later
                x_min = 10
                x_max = 110
            else:
                # For numeric scores, use original logic
                dataframe[y_column] = pd.to_numeric(dataframe[y_column], errors="coerce")
                clean_df = dataframe.dropna(subset=[x_column, y_column])
                clean_df = clean_df.sort_values(by=y_column, ascending=True)
                df_median = clean_df[[x_column, y_column]].groupby(y_column).median()

                # Set proper limits based on score range
                if eval_score_range == "1_to_10":
                    x_min = 0.5
                    x_max = 10.5
                elif eval_score_range == "0_to_1":
                    x_min = -0.05
                    x_max = 1.05
                elif eval_score_range == "-1_to_1":
                    x_min = -1.05
                    x_max = 1.05
                else:
                    # Fallback to data-based limits
                    x_min = clean_df[x_column].min() - 1
                    x_max = clean_df[x_column].max() + 1
        
            # Calculate errors based on the appropriate column
            if eval_score_range == "A_to_E":
                lower_errors, upper_errors = self.calculate_errors(clean_df, y_column, x_column_numeric)
            else:
                lower_errors, upper_errors = self.calculate_errors(clean_df, y_column, x_column)

            fig, ax = plt.subplots(figsize=(16, 10), dpi=80)

            for i, (idx, row) in enumerate(df_median.iterrows()):
                # Use appropriate column for plotting
                plot_column = x_column_numeric if eval_score_range == "A_to_E" else x_column
                
                if error_bars:
                        if swap_axes:
                            ax.errorbar(
                                y=df_median.loc[idx, plot_column], x=i,
                                yerr=[[lower_errors[idx]], [upper_errors[idx]]],  # Vertical error
                                fmt='o', color='firebrick',
                                capsize=5  # Set a visible cap size
                            )
                        else:
                            ax.errorbar(
                                x=df_median.loc[idx, plot_column], y=i,
                                xerr=[[lower_errors[idx]], [upper_errors[idx]]],  # Horizontal error
                                fmt='o', color='firebrick',
                                capsize=5  # Set a visible cap size
                        )
                if not show_only_medians:
                    df_category = clean_df[clean_df[y_column] == idx]

                    if swap_axes:
                        ax.scatter(
                            y=df_category[plot_column],  # Use appropriate column values
                            x=np.repeat(i, df_category.shape[0]),
                            s=circle_size,
                            edgecolors="gray",
                            color=(1, 0.5, 0, 0.5),
                            alpha=0.3,
                        )
                    else:
                        ax.scatter(
                            x=df_category[plot_column],  # Use appropriate column values
                            y=np.repeat(i, df_category.shape[0]),
                            s=circle_size,
                            edgecolors="gray",
                            color=(1, 0.5, 0, 0.5),
                            alpha=0.3,
                        )

                if swap_axes:
                    ax.scatter(
                        y=df_median.loc[idx, plot_column], x=i, s=circle_size, c="grey"
                    )
                else:
                    ax.scatter(
                        x=df_median.loc[idx, plot_column], y=i, s=circle_size, c="grey"
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
                
            # Set proper Y-axis limits and ticks based on the score range
            if eval_score_range == "A_to_E":
                if swap_axes:
                    ax.set_ylim(10, 110)  # A-E range is 20-100%
                    ax.set_yticks([20, 40, 60, 80, 100])
                    ax.set_yticklabels(['E', 'D', 'C', 'B', 'A'])
                else:
                    ax.set_xlim(10, 110)  # A-E range is 20-100%
                    ax.set_xticks([20, 40, 60, 80, 100])
                    ax.set_xticklabels(['E', 'D', 'C', 'B', 'A'])
            elif eval_score_range == "1_to_10":
                if swap_axes:
                    ax.set_ylim(0.5, 10.5)  # 1-10 range
                else:
                    ax.set_xlim(0.5, 10.5)  # 1-10 range
            elif eval_score_range == "0_to_1":
                if swap_axes:
                    ax.set_ylim(-0.05, 1.05)  # 0-1 range
                else:
                    ax.set_xlim(-0.05, 1.05)  # 0-1 range
            elif eval_score_range == "-1_to_1":
                if swap_axes:
                    ax.set_ylim(-1.05, 1.05)  # -1 to 1 range
                else:
                    ax.set_xlim(-1.05, 1.05)  # -1 to 1 range

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

            ax.set_title(f"{error_mode.title()} Data", fontdict={"size": 18})

            plt.xticks(alpha=0.7)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["bottom"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.grid(axis="both", alpha=0.4, linewidth=0.1)

            output_png_path = run_name + "_graph.png"
            plt.savefig(output_png_path, bbox_inches="tight")

            plt.show()

    def plot_violin_distribution(
            self,
            dataframe,
            x_column,
            y_column,
            run_name,
            error_mode,
            model_name="",
            eval_score_range="1_to_10",
        ):
            """Create violin plots to show the distribution of scores across different error percentages."""
            # Handle letter grades differently
            if eval_score_range == "A_to_E":
                # For letter grades, we need to convert to numeric for plotting
                # Mapping: A=100%, B=80%, C=60%, D=40%, E=20%
                letter_to_num = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}
                dataframe = dataframe.copy()
                dataframe[x_column + '_numeric'] = dataframe[x_column].map(letter_to_num)
                x_column_numeric = x_column + '_numeric'
                
                # Use numeric column for calculations
                dataframe[y_column] = pd.to_numeric(dataframe[y_column], errors="coerce")
                clean_df = dataframe.dropna(subset=[x_column_numeric, y_column])
                
                # Create violin plot with numeric values
                fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
                
                # Group by error percentage and create violin plot
                violin_data = [clean_df[clean_df[y_column] == perc][x_column_numeric].values 
                              for perc in sorted(clean_df[y_column].unique())]
                violin_labels = [f"{perc:.1f}%" for perc in sorted(clean_df[y_column].unique())]
                
                # Create violin plot
                parts = ax.violinplot(violin_data, positions=range(len(violin_data)), 
                                    showmeans=True, showmedians=True)
                
                # Customize violin plot appearance
                parts['cmeans'].set_color('red')
                parts['cmeans'].set_linewidth(2)
                parts['cmedians'].set_color('blue')
                parts['cmedians'].set_linewidth=2
                
                # Set labels and title
                ax.set_xlabel("Error Percentage", alpha=0.7)
                ax.set_ylabel("Score", alpha=0.7)
                ax.set_title(f"{error_mode.title()} Data - Violin Plot", fontdict={"size": 18})
                
                # Set x-axis ticks
                ax.set_xticks(range(len(violin_labels)))
                ax.set_xticklabels(violin_labels)
                
                # Set y-axis limits and ticks for letter grades
                ax.set_ylim(10, 110)  # A-E range is 20-100%
                ax.set_yticks([20, 40, 60, 80, 100])
                ax.set_yticklabels(['E', 'D', 'C', 'B', 'A'])
                
                # Add grid
                ax.grid(axis="both", alpha=0.4, linewidth=0.1)
                
            else:
                # For numeric scores, use original logic
                dataframe[y_column] = pd.to_numeric(dataframe[y_column], errors="coerce")
                clean_df = dataframe.dropna(subset=[x_column, y_column])
                
                # Create violin plot
                fig, ax = plt.subplots(figsize=(16, 10), dpi=80)
                
                # Group by error percentage and create violin plot
                violin_data = [clean_df[clean_df[y_column] == perc][x_column].values 
                              for perc in sorted(clean_df[y_column].unique())]
                violin_labels = [f"{perc:.1f}%" for perc in sorted(clean_df[y_column].unique())]
                
                # Create violin plot
                parts = ax.violinplot(violin_data, positions=range(len(violin_data)), 
                                    showmeans=True, showmedians=True)
                
                # Customize violin plot appearance
                parts['cmeans'].set_color('red')
                parts['cmeans'].set_linewidth(2)
                parts['cmedians'].set_color('blue')
                parts['cmedians'].set_linewidth=2
                
                # Set labels and title
                ax.set_xlabel("Error Percentage", alpha=0.7)
                ax.set_ylabel("Score", alpha=0.7)
                ax.set_title(f"{error_mode.title()} Data - Violin Plot", fontdict={"size": 18})
                
                # Set x-axis ticks
                ax.set_xticks(range(len(violin_labels)))
                ax.set_xticklabels(violin_labels)
                
                # Set y-axis limits based on score range
                if eval_score_range == "1_to_10":
                    ax.set_ylim(0.5, 10.5)  # 1-10 range
                elif eval_score_range == "0_to_1":
                    ax.set_ylim(-0.05, 1.05)  # 0-1 range
                elif eval_score_range == "-1_to_1":
                    ax.set_ylim(-1.05, 1.05)  # -1 to 1 range
                
                # Add grid
                ax.grid(axis="both", alpha=0.4, linewidth=0.1)
            
            # Remove spines
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            
            output_png_path = run_name + "_violin.png"
            plt.savefig(output_png_path, bbox_inches="tight")
            print(f"Violin plot saved as: {output_png_path}")
            
            plt.show()

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
        print("Starting test for score based Evals...")
        print(f"- Model: {self.model_name}")
        print(f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print(f"- Document Depths: {len(self.document_error_percents)}, Min: {min(self.document_error_percents)}%, Max: {max(self.document_error_percents)}%")
        print("Error Mode " + self.error_mode)
        print("Eval Score Range " + self.eval_score_range)
        print("Model Provider " + self.model_provider)
        print("Number of statistical runs per range " + str(self.number_of_runs_per_context_length))
        print("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()

    def run_both_tests(self, model_provider="OpenAI", model_name=None, eval_score_range="1_to_10"):
        """Run tests for frustration, sadness, and spelling errors modes"""
        print(f"Running tests for frustration, sadness, and spelling errors using {model_provider}...")
        print(f"Score range: {eval_score_range}")
        
        if model_name is None:
            if model_provider == "OpenAI":
                model_name = "gpt-5-nano"
                # model_name = "gpt-4"
                # model_name = "o3"
            elif model_provider == "Anthropic":
                model_name = "claude-opus-4-20250514"
                # model_name = "claude-sonnet-4-20250514",
            elif model_provider == "Qwen":
                model_name = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"
        
        print("\n" + "="*50)
        print(f"TESTING FRUSTRATION MODE with {model_provider}:{model_name}")
        print("="*50)
        frustration_tester = LLMNumericScoreEvalTester(
            error_mode="frustration",
            model_provider=model_provider,
            model_name=model_name,
            eval_score_range=eval_score_range
        )
        frustration_tester.start_test()
        
        plt.close('all')
        
        print("\n" + "="*50)
        print(f"TESTING SADNESS MODE with {model_provider}:{model_name}")
        print("="*50)
        sadness_tester = LLMNumericScoreEvalTester(
            error_mode="sadness",
            model_provider=model_provider,
            model_name=model_name,
            eval_score_range=eval_score_range
        )
        sadness_tester.start_test()
        
        plt.close('all')
        
        print("\n" + "="*50)
        print(f"TESTING SPELLING ERRORS MODE with {model_provider}:{model_name}")
        print("="*50)
        spelling_tester = LLMNumericScoreEvalTester(
            error_mode="spelling_errors",
            model_provider=model_provider,
            model_name=model_name,
            eval_score_range=eval_score_range
        )
        spelling_tester.start_test()


if __name__ == "__main__":
    ''' Available options:
    MODEL_PROVIDER = "OpenAI", MODEL_NAME = "gpt-4", "o3", "gpt-5-mini", etc.
    MODEL_PROVIDER = "Anthropic", MODEL_NAME = "claude-opus-4-20250514", "claude-sonnet-4-20250514", etc.
    MODEL_PROVIDER = "Qwen", MODEL_NAME = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo", etc.
    EVAL_SCORE_RANGE = "1_to_10", "0_to_1", "-1_to_1", "A_to_E" '''
    
    MODEL_PROVIDER = "OpenAI"
    MODEL_NAME = "gpt-4"
    # MODEL_NAME = "gpt-5-nano"
    # MODEL_NAME = "o3"

    # MODEL_PROVIDER = "Anthropic"
    # MODEL_NAME = "claude-opus-4-20250514"
    # MODEL_NAME = "claude-sonnet-4-20250514"

    # MODEL_PROVIDER = "Qwen"
    # MODEL_NAME = "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo"
    # MODEL_NAME = "fireworks/accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"

    EVAL_SCORE_RANGE =  "1_to_10"
    project_name = "Binary vs Eval Score Test:"

    tracer_provider = register(project_name=project_name, auto_instrument=True)
    tracer = tracer_provider.get_tracer(__name__)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    
    ht = LLMNumericScoreEvalTester()
    ht.run_both_tests(model_provider=MODEL_PROVIDER, model_name=MODEL_NAME, eval_score_range=EVAL_SCORE_RANGE)