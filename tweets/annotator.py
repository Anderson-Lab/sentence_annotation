import os.path
import pandas as pd
from ipywidgets import Label, Button, Layout, HBox
from IPython.display import display, clear_output


COLUMNS = ['user_id', 'tweet_id', 'sentiment', 'tweet', 'words', 'mask']


def init_original_df(file_name, sentiments=['"negative"', '"positive"', '"neutral"']):
    fd = open(file_name, 'r')
    lines = fd.readlines()
    fd.close()
    splits = []
    for line in lines:
        split = line.split('\t')[:4]
        split.append(split[-1].split())
        split.append([])
        splits.append(split)
    df = pd.DataFrame(splits, columns=COLUMNS)
    df = df[df.sentiment.isin(sentiments)]
    return df


def init_mask_df(file_name):
    if os.path.isfile(file_name):
        return pd.read_pickle(file_name)
    else:
        return pd.DataFrame([], columns=COLUMNS)


def save_mask_df(mask_df, file_name):
    mask_df.to_pickle(file_name)


def remove_already_seen(original_df, seen_df):
    return original_df[~original_df.index.isin(seen_df.index.tolist())]


def get_random_tweet(df, sentiments=['"negative"', '"positive"', '"neutral"']):
    # subset to rows that do not already have a mask
    new_df = df[(df['mask'].str.len() == 0) & (
        df['sentiment'].isin(sentiments))].reset_index()
    return new_df.sample(n=1)


def on_create_mask(row, mask, mask_file_name, sentiments):
    global original_df
    global mask_df
    global unseen_df
    row['mask'] = [mask]
    mask_df = mask_df.append(row)
    save_mask_df(mask_df, mask_file_name)
    reset_display(mask_file_name, sentiments)


def display_buttons(row, create_mask_callback, mask_file_name, sentiments=['"negative"', '"positive"', '"neutral"']):
    global mask_df
    sentiment = row.sentiment.tolist()[0]
    words = row.words.tolist()[0]
    sentiment_text = Label(value=sentiment)
    sentiment_wrapper = HBox([sentiment_text], layout=Layout(
        margin='15px', justify_content='center'))
    display(sentiment_wrapper)
    buttons = []
    for word in words:
        layout = Layout(width='auto', height='auto', margin='4px')
        button = Button(description=word, layout=layout)

        def on_click(b):
            if b.style.button_color is not None:
                b.style.button_color = None
            else:
                b.style.button_color = '#66CC69'
        button.on_click(on_click)
        buttons.append(button)
    wrapper = HBox(buttons, layout=Layout(
        flex_flow='row wrap', justify_content='center'))
    display(wrapper)
    create_button = Button(description='Create Mask', layout=Layout(
        width='auto', height='auto', padding='2px 15px 2px 15px', margin='10px'))
    create_button.style.button_color = '#5EACF9'
    skip_button = Button(description='Skip Tweet', layout=Layout(
        width='auto', height='auto', padding='2px 15px 2px 15px', margin='10px'))
    skip_button.style.button_color = '#F9DC5C'
    def create_callback(b):
        mask = []
        for button in buttons:
            mask.append(button.style.button_color is not None)
        clear_output()
        create_mask_callback(row, mask, mask_file_name, sentiments)
    def skip_callback(b):
        clear_output()
        reset_display(mask_file_name, sentiments)
    create_button.on_click(create_callback)
    skip_button.on_click(skip_callback)
    create_wrapper = HBox([create_button, skip_button], layout=Layout(
        margin='20px', justify_content='center'))
    display(create_wrapper)
    done_texts = []
    for sent in sentiments:
        new_df = mask_df[mask_df.sentiment == sent]
        done_texts.append(Label(value="%s: %s" % (sent, str(len(new_df))), layout=Layout(margin='20px')))
    done_wrapper = HBox(done_texts)
    display(done_wrapper)


def reset_display(mask_file_name, sentiments):
    global original_df
    global mask_df
    global unseen_df
    unseen_df = remove_already_seen(original_df, mask_df)
    tweet = get_random_tweet(unseen_df, sentiments)
    display_buttons(tweet, on_create_mask, mask_file_name, sentiments)


def run_annotator(original_file_name, mask_file_name, sentiments=['"negative"', '"positive"', '"neutral"']):
    global original_df
    global mask_df
    global unseen_df
    original_df = init_original_df(original_file_name)
    mask_df = init_mask_df(mask_file_name)
    reset_display(mask_file_name, sentiments)
