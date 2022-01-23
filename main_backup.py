import PySimpleGUI as sg
import os


def make_window(theme):
    menu_def = [["&Application", ["E&xit"]], ["&Help", ["&About"]]]
    right_click_menu_def = [
        [],
        ["Edit Me", "Versions", "Nothing", "More Nothing", "Exit"],
    ]
    graph_right_click_menu_def = [
        [],
        ["Erase", "Draw Line", "Draw", ["Circle", "Rectangle", "Image"], "Exit"],
    ]

    # Table Data
    data = [["John", 10], ["Jen", 5]]
    headings = ["Name", "Score"]

    input_layout = [
        # [sg.Menu(menu_def, key='-MENU-')],
        [sg.Text(".env Config")],
        [sg.Checkbox("Load CLIP?", default=True, k="-CB1-")],
        [sg.Checkbox("Load Indices?", default=True, k="-CB2-")],
        [sg.Checkbox("Use Dev?", default=True, k="-CB3-")],
        [
            sg.Combo(
                values=(
                    "vqgan_imagenet_f16_16384",
                    "vqgan_faceshq",
                    "vqgan_imagenet_f16_1024",
                    "wikiart_16384",
                ),
                default_value="Model",
                readonly=True,
                k="-COMBO1-",
            ),
            sg.Combo(
                values=("412", "288", "512", "-------"),
                default_value="Width",
                readonly=True,
                k="-COMBO2-",
            ),
            sg.Combo(
                values=("412", "288", "512", "-------"),
                default_value="Height",
                readonly=True,
                k="-COMBO3-",
            ),
            sg.Combo(
                values=("400", "500", "600", "-------"),
                default_value="Iterations",
                readonly=True,
                k="-COMBO4-",
            ),
            sg.Combo(
                values=("125", "150", "175", "200", "-------"),
                default_value="Diffusion",
                readonly=True,
                k="-COMBO5-",
            ),
        ],
        [sg.Button("Save"), sg.Button("Start")],
    ]

    logging_layout = [
        [sg.Text("Anything printed will display here!")],
        [
            sg.Multiline(
                size=(60, 15),
                font="Courier 8",
                expand_x=True,
                expand_y=True,
                write_only=True,
                reroute_stdout=True,
                reroute_stderr=True,
                echo_stdout_stderr=True,
                autoscroll=True,
                auto_refresh=True,
            )
        ]
        # [sg.Output(size=(60,15), font='Courier 8', expand_x=True, expand_y=True)]
    ]

    layout = [
        [sg.MenubarCustom(menu_def, key="-MENU-", font="Courier 15", tearoff=True)],
        [
            sg.Text(
                "BATbot Startup",
                size=(38, 1),
                justification="center",
                font=("Helvetica", 16),
                relief=sg.RELIEF_RIDGE,
                k="-TEXT HEADING-",
                enable_events=True,
            )
        ],
    ]
    layout += [
        [
            sg.TabGroup(
                [
                    [
                        sg.Tab("Input Elements", input_layout),
                        sg.Tab("Output", logging_layout),
                    ]
                ],
                key="-TAB GROUP-",
                expand_x=True,
                expand_y=True,
            ),
        ]
    ]
    layout[-1].append(sg.Sizegrip())
    window = sg.Window(
        "BATbot Startup",
        layout,
        right_click_menu=right_click_menu_def,
        right_click_menu_tearoff=True,
        grab_anywhere=True,
        resizable=True,
        margins=(0, 0),
        use_custom_titlebar=True,
        finalize=True,
        keep_on_top=True,
        # scaling=2.0,
    )
    window.set_min_size(window.size)
    return window


def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
            print("============ Event = ", event, " ==============")
            print("-------- Values Dictionary (key=value) --------")
            for key in values:
                print(key, " = ", values[key])
            os.environ["model"] = values["-COMBO1-"]
            print(os.environ["model"])
            os.environ["width"] = values["-COMBO2-"]
            print(os.environ["width"])
            os.environ["height"] = values["-COMBO3-"]
            print(os.environ["height"])
            os.environ["max_iterations"] = values["-COMBO4-"]
            print(os.environ["max_iterations"])
            os.environ["diffusion_iterations"] = values["-COMBO5-"]
            print(os.environ["diffusion_iterations"])
            os.environ["CB1"] = str(values["-CB1-"])
            os.environ["CB2"] = str(values["-CB2-"])
            os.environ["CB3"] = str(values["-CB3-"])
        if event in (None, "Exit"):
            print("[LOG] Clicked Exit!")
        elif event == "Start":
            print("[LOG] Clicked Popup Button!")
            sg.popup("BATbot is on!", keep_on_top=True)
            window.close()
            os.system("python3 Bot/bot.py")
        elif event == "Edit Me":
            sg.execute_editor(__file__)
        elif event == "Versions":
            sg.popup(sg.get_versions(), keep_on_top=True)


if __name__ == "__main__":
    sg.theme("DefaultNoMoreNagging")
    main()
