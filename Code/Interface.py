import tkinter as tk
from tkinter import *
import  tkinter.messagebox
#import weka.core.jvm as jvm
import joblib

ml_models = ('lasso_for.sav', 'lasso_against.sav', 'lasso_positive.sav', 'linearRegression_negative.sav','lasso_factual.sav', 'lasso_emotional.sav' )
labels = ('Pro value', 'Contra value', 'Positive sentiment value', 'Negative sentiment value','Evidence-based value', 'Emotion-based value' )
explanations = {
    'Pro value': """Implies that the text contains statements or arguments supporting the use of vaccinations. They can range from simply advertising places where one can get vaccinations, to openly promoting the benefits of vaccinations.""",
    'Contra value': """Implies that the text contains statements or arguments opposing the use of vaccinations. They can range from slight uncertainty about them, to openly promoting that vaccinations are bad and that nobody should be getting them.""",
    'Positive sentiment value': """Implies that some aspects of the tweet uncover a positive mood, such as praise, recommendations or a favorable comparison.""",
    'Negative sentiment value': """Implies that some aspects of the tweet uncover a negative mood such as, criticism, insults or a negative comparison.""",
    'Evidence-based value': """Judges how based on evidence the argument(s)/statement(s) made in the tweet are, e.g. whether some organization was quoted, whether statistics and/or scientific findings were used, regardless of personal belief of the evidence used regarding
    accuracy and reliability.""",
    'Emotion-based value': """Judges how based on emotions the argument(s)/statement(s) made in the tweet are, e.g. whether the expressions and language used was meant to cause certain emotions in the reader, and/or whether they reflected the author's own emotions. For example, the statement \"get your kids vaccinated, otherwise they'll die horrible deaths\", is appealing strongly to emotions."""
}

def input_screen(root):
    clear_window(root)
    ents = make_form(root)
    b1 = tk.Button(root, text='Label tweet',
           command=(lambda e=ents: label_tweets(e, root)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)


def make_form(root):
    row = tk.Frame(root)
    lab = tk.Label(row,  text="Input tweet text:", anchor='w')
    text = tk.Text(row)

    text.insert(tk.END, "Copy tweet text here")
    row.pack(side=tk.TOP,
             fill=tk.X,
             padx=5,
             pady=5)
    lab.pack(side=tk.TOP)
    text.pack()
    return text

def label_tweets(textBox, root):
    text = retrieve_input(textBox)
#    print(text)
#    loaded_model = joblib.load('ridgeRegressionTweet.sav')
    clear_window(root)
    tweet = tk.Label(root, text=text, font=("Verdana", 11), wraplength=700, relief=RIDGE)
    tweet.pack(side=tk.TOP, fill=tk.X, pady=15)
    text = [text]
#    print(loaded_model.predict(text))

    for i in range(6):
        label = labels[i]
        ml_model = ml_models[i]
        loaded_model = joblib.load(ml_model)
        # Replace value with actual value
        value = float(loaded_model.predict(text))
        row = tk.Frame(root)
        txt = tk.Frame(row)
        lab = tk.Label(txt, justify=tk.LEFT,font=("Verdana bold", 10), text=label+": "+str(value))




        info_b = tk.Label(txt, text="?", relief=RAISED)
        info = tk.Label(root, text=explanations[label], wraplength=400, relief=GROOVE)

        info.place(relx=0.5,rely=0.5,anchor=CENTER)
        info.lower()

        #info = tk.Button(txt, text='?',
        #           command=(lambda: tk.messagebox.showinfo('Info', explanations[label])))
        #desc = tk.Label(txt, justify=tk.LEFT,wraplength=500, font=("Verdana", 7), text=explanations[label])
        scale = tk.Scale(row, from_=0, to=5, orient=HORIZONTAL)
        scale.set(value)


        row.pack(side=tk.TOP, fill= tk.X)
        txt.pack(side=tk.LEFT)
        lab.pack(side=tk.LEFT)
        info_b.pack(side=tk.RIGHT)
        info_b.bind("<Enter>", lambda e=info: info.lift())
        info_b.bind("<Leave>", lambda e=info: info.lower())
        info.bind("<Enter>", lambda e=info: info.lift())
        info.bind("<Leave>", lambda e=info: info.lower())
        #info.pack(side=tk.RIGHT, padx=5, pady=5)
        #desc.pack(side=tk.BOTTOM )
        scale.pack(side=tk.RIGHT,fill=tk.X)
#        print(scale.get())

    b1 = tk.Button(root, text='Label new tweet',
           command=(lambda: input_screen(root)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)

    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)


def retrieve_input(text):
    inputValue=text.get("1.0","end-1c")
    return inputValue

def clear_window(root):
    list = root.pack_slaves()
    for l in list:
        l.destroy()

if __name__ == '__main__':
#    jvm.start(packages="/Users/baoziyu/wekafiles", max_heap_size="512m")

    root = tk.Tk()
    input_screen(root)
    
    root.title("Tweet Labeler")
    root.mainloop()
    
#    jvm.stop()
