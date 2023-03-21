import os
import tiktoken
import tkinter as tk
import pinecone
import customtkinter
import tkinter as tk
from ttkthemes import ThemedStyle

import tkinter as tk
import customtkinter


class ChatWindow(customtkinter.CTkTextbox):
    """A custom textbox widget that behaves like a chat window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag_config("user", foreground="blue", justify="right")
        self.tag_config("bot", foreground="green")
        self.configure(state="disabled", font=("Helvetica", 14))

    def add_message(self, message, sender):
        """Adds a message to the chat history."""
        self.configure(state="normal")
        self.insert(tk.END, sender + ": " + message + "\n", sender)
        self.see(tk.END)
        self.configure(state="disabled")


def send_message():
    """A sample function to send a message."""
    message = input_field.get()
    chat_history.add_message(message, "user")
    # Do some processing with the message here...
    chat_history.add_message("This is a sample bot response.", "bot")
    input_field.delete(0, tk.END)


root = customtkinter.CTk()
root.title("Chat")

# Set the background color of the root window
root.configure(bg="#9A8C98")

customtkinter.set_appearance_mode("dark")

# Creating Chat History
chat_history = ChatWindow(root, height=500, width=400) 
chat_history.pack(side=tk.TOP, padx=20, pady=20, fill=tk.BOTH, expand=True)

# Creating Input Field
input_frame = customtkinter.CTkFrame(root)
input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20)

input_field = customtkinter.CTkEntry(input_frame, font=("Helvetica", 14))
input_field.pack(side=tk.LEFT, padx=10, pady=10, ipady=8, fill=tk.X, expand=True)

send_button = customtkinter.CTkButton(input_frame, text="Send", font=("Helvetica", 14), command=send_message)
send_button.pack(side=tk.LEFT, padx=10, pady=10, ipady=7, ipadx=15)

# Disable editing of the chat history
chat_history.add_message("Welcome to the chat app!", "bot")

root.mainloop()








