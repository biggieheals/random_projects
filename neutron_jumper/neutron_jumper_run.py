import json
import csv
import os
import numpy as np
from PIL import ImageGrab
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import keyboard
import pyautogui
import pyperclip
import easyocr
import time
import pandas as pd  # Using pandas for CSV filtering

# --------------------- EASYOCR SETUP ---------------------
reader = easyocr.Reader(['en'], gpu=False)

# --------------------- FILE LOADING ---------------------
def load_route(path="jump_data.csv"):
    """
    Load route data from a JSON or CSV file.
    For CSV files, this function isn't used (pandas is used instead).
    It is used if a JSON file is selected.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["jumps"]
    elif ext == ".csv":
        jumps = []
        with open(path, "r", encoding="utf-8") as f:
            reader_csv = csv.DictReader(f)
            for row in reader_csv:
                jumps.append(row)
        return jumps
    else:
        raise ValueError("Unsupported file format. Please use .json or .csv")

# --------------------- GALAXY MAP CONTROL ---------------------
def open_galaxy_map_and_plot_route():
    """
    1) Press F11 to trigger your AHK hotkey (assuming AHK is running).
    2) Wait a bit so Elite's galaxy map can open.
    """
    pyautogui.press('f11')  # triggers your existing AHK hotkey
    time.sleep(15)
    return

# --------------------- GUI CLASS ---------------------
class NeutronJumpGUI:
    def __init__(self, master, route_data):
        """
        Build the GUI. Also skip the first system (index 0) in the route_data,
        because that's the one you're already in and don't need to plot.
        """
        # Remove the first system if there's at least one row
        if len(route_data) > 1:
            route_data = route_data[1:]  # skip the first row entirely

        self.master = master
        self.route_data = route_data
        self.total_jumps = len(route_data)
        self.current_jump = 0

        self.master.title("Neutron Jump Overlay")
        self.master.geometry("650x400+10+10")
        self.master.attributes("-topmost", True)
        self.master.configure(bg='black')

        self.title_label = tk.Label(master, text="Neutron Jump Route",
                                    bg='black', fg='white',
                                    font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=5)

        self.progress_label = tk.Label(master, text="Progress: 0 / 0",
                                       bg='black', fg='white',
                                       font=("Helvetica", 12, "bold"))
        self.progress_label.pack(pady=5)

        self.current_system_label = tk.Label(master, text="Current System: ---",
                                             bg='black', fg='yellow',
                                             font=("Helvetica", 12, "bold"))
        self.current_system_label.pack(pady=5)

        # Determine the columns from the route data; add a 'Status' column if missing
        if self.total_jumps > 0:
            self.columns = list(route_data[0].keys())
        else:
            self.columns = ["System Name"]
        if "Status" not in self.columns:
            self.columns.append("Status")

        self.tree = ttk.Treeview(master, columns=self.columns, show='headings', height=10)
        for col in self.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor='center', stretch=True)
        self.tree.pack(padx=10, pady=5, fill='x', expand=True)

        def format_value(val):
            try:
                num = float(val)
                return f"{num:.1f}"
            except (ValueError, TypeError):
                return val

        # Populate table – if a row doesn’t have a "Status" field, default to "Pending"
        for row in route_data:
            row_values = [format_value(row.get(col, "Pending")) for col in self.columns]
            self.tree.insert("", "end", values=row_values)

        self.button = tk.Button(master, text="Next Jump",
                                command=self.next_jump, bg='darkgreen', fg='white')
        self.button.pack(pady=10)

        # Bind F10 for next jump and Shift+F10 for reattempting the same jump
        threading.Thread(target=self.hotkey_listener, daemon=True).start()
        self.update_progress_label()

    def hotkey_listener(self):
        """
        Bind F10 to trigger the next jump and Shift+F10 to reattempt the current plot.
        """
        keyboard.add_hotkey("f10", self.next_jump)
        keyboard.add_hotkey("shift+f10", self.reattempt_jump)
        keyboard.wait()

    def next_jump(self):
        if self.current_jump >= self.total_jumps:
            messagebox.showinfo("Route Complete", "You have completed all jumps!")
            return

        # Get the row data for the upcoming jump
        row_data = self.route_data[self.current_jump]
        # Try to get system name from "System Name" or fallback to first column's value
        system_name = row_data.get("System Name", None)
        if system_name is None:
            system_name = list(row_data.values())[0]

        # Copy the system name to clipboard so AHK's Ctrl+V can paste it
        pyperclip.copy(system_name)

        # Update GUI label
        self.current_system_label.config(text=f"Plotting route to: {system_name}")

        # Run the jump routine in a separate thread
        threading.Thread(target=self.execute_jump, args=(system_name,), daemon=True).start()

    def execute_jump(self, system_name):
        """
        1) Trigger the galaxy map (via F11, which your AHK script handles).
        2) Mark the jump as completed in the GUI table.
        3) Update the label to show the next jump info.
        """
        open_galaxy_map_and_plot_route()

        # Update the tree row's status
        tree_children = self.tree.get_children()
        if self.current_jump < len(tree_children):
            row_id = tree_children[self.current_jump]
            new_status = f"Plotted route to {system_name}"
            self.tree.set(row_id, "Status", new_status)
            self.tree.item(row_id, tags=("done",))
            self.tree.tag_configure("done", background="gray")

        # Determine the next system name if available
        next_system_name = "Route complete!"
        if self.current_jump + 1 < len(self.route_data):
            next_row = self.route_data[self.current_jump + 1]
            next_system_name = next_row.get("System Name") or list(next_row.values())[0]

        # Update label with current and next jump info
        self.current_system_label.config(
            text=f"Plotted route to {system_name}. F10 when neutron charged to plot next system: {next_system_name}"
        )

        # Increment jump index and update progress label
        self.current_jump += 1
        self.update_progress_label()

    def reattempt_jump(self):
        """
        Reattempt plotting the current jump without advancing the jump index.
        This method is triggered by Shift+F10.
        """
        if self.current_jump >= self.total_jumps:
            messagebox.showinfo("Route Complete", "You have completed all jumps!")
            return

        # Get the current row data (same as in next_jump)
        row_data = self.route_data[self.current_jump]
        system_name = row_data.get("System Name", None)
        if system_name is None:
            system_name = list(row_data.values())[0]

        # Copy the system name to clipboard so AHK's Ctrl+V can paste it
        pyperclip.copy(system_name)

        # Update label to indicate reattempt
        self.current_system_label.config(text=f"Reattempting plot for: {system_name}")

        # Run the galaxy map routine again
        open_galaxy_map_and_plot_route()

        # Optionally update the tree row's status to reflect reattempt
        tree_children = self.tree.get_children()
        if self.current_jump < len(tree_children):
            row_id = tree_children[self.current_jump]
            new_status = f"Reattempted plot to {system_name}"
            self.tree.set(row_id, "Status", new_status)
            self.tree.item(row_id, tags=("reattempt",))
            self.tree.tag_configure("reattempt", background="orange")

        # Determine the next system name if available (for display purposes)
        next_system_name = "Route complete!"
        if self.current_jump + 1 < len(self.route_data):
            next_row = self.route_data[self.current_jump + 1]
            next_system_name = next_row.get("System Name") or list(next_row.values())[0]

        # Update label with next system info (without advancing the jump)
        self.current_system_label.config(
            text=f"Reattempted plot to {system_name}. F10 when neutron charged to plot next system: {next_system_name}"
        )

    def update_progress_label(self):
        self.progress_label.config(text=f"Progress: {self.current_jump} / {self.total_jumps}")

# --------------------- ENTRYPOINT ---------------------
def main():
    # Create a temporary Tk instance for dialogs
    root = tk.Tk()
    root.withdraw()

    # Use Tkinter's file dialog to select the jump data file
    file_path = filedialog.askopenfilename(
        title="Select jump data file",
        filetypes=(("CSV files", "*.csv"), ("JSON files", "*.json"), ("All Files", "*.*"))
    )
    if not file_path:
        messagebox.showerror("Error", "No file selected!")
        return

    # Load data depending on file type
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
        # Ask via a yes/no dialog if the route includes normal jumps.
        # Yes: Filter out rows where both "Refuel" and "Neutron Star" are "No"
        route_includes_normal = messagebox.askyesno(
            "Normal Jumps?",
            "Does your route include normal jumps?\n\n"
            "Yes: Filter out rows where both 'Refuel' and 'Neutron Star' are 'No'.\n"
            "No: Use all rows (neutrons only)."
        )
        if route_includes_normal:
            df = df[~((df["Refuel"].str.lower() == "no") & (df["Neutron Star"].str.lower() == "no"))]
        route_data = df.to_dict(orient="records")
    elif file_path.lower().endswith(".json"):
        route_data = load_route(file_path)
    else:
        messagebox.showerror("Error", "Unsupported file type!")
        return

    # Destroy the temporary root window and launch the main GUI
    root.destroy()
    main_root = tk.Tk()
    gui = NeutronJumpGUI(main_root, route_data)
    main_root.mainloop()

if __name__ == "__main__":
    main()
