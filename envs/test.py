# 계산기를 만들건데 GUI를 이용해서 만들어야 해

from tkinter import *


# 버튼을 눌렀을 때 동작하는 함수
def click(key):
    # = 버튼이 눌렸을 때 계산 수행
    if key == '=':
        try:
            result = str(eval(display.get()))[0:10]
            display.insert(END, " = " + result)
        except:
            display.insert(END, " --> Error!")

    # C 버튼이 눌렸을 때 display 엔트리 위젯 내용 비움
    elif key == 'C':
        display.delete(0, END)

    # 그 외의 경우, 버튼 눌릴 때마다 display 엔트리 위젯의 내용에 추가
    else:
        display.insert(END, key)


# 메인 코드 부분
window = Tk()
window.title("My Calculator")
display = Entry(window, width=33, bg="yellow")
display.grid(row=0, column=0, sticky="w", columnspan=5)

# 버튼을 위한 버튼 와리아블 준비
button_list = [
    '7', '8', '9', '/', 'C',
    '4', '5', '6', '*', ' ',
    '1', '2', '3', '-', ' ',
    '0', '.', '=', '+', ' ']

# 버튼을 위한 반복문
row_index = 1
col_index = 0
for button_text in button_list:

    def process(t=button_text):
        click(t)


    Button(window, text=button_text, width=5, command=process).grid(row=row_index, column=col_index)
    col_index += 1
    if col_index > 4:
        row_index += 1
        col_index = 0

window.mainloop()

import tkinter as tk


class Calculator(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.display = tk.Entry(self)
        self.display.grid(row=0, column=0, columnspan=4)
        self.display.insert(0, "0")

        self.seven = tk.Button(self, text="7", command=lambda: self.add_to_display("7"))
        self.seven.grid(row=1, column=0)
        self.eight = tk.Button(self, text="8", command=lambda: self.add_to_display("8"))
        self.eight.grid(row=1, column=1)
        self.nine = tk.Button(self, text="9", command=lambda: self.add_to_display("9"))
        self.nine.grid(row=1, column=2)
        self.divide = tk.Button(self, text="/", command=lambda: self.add_to_display("/"))
        self.divide.grid(row=1, column=3)

    def add_to_display(self, char):
        self.display.insert(len(self.display.get()), char)


root = tk.Tk()
