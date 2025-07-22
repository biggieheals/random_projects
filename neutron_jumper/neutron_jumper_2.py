#Requires Python 3.x
import json
import csv
import os
import numpy as np
# from PIL import ImageGrab # Not used in the provided snippet
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Scrollbar # Added Scrollbar
import threading
import keyboard
import pyautogui
import pyperclip
# import easyocr # Not used directly in the GUI part, assumed setup elsewhere
import time
import pandas as pd

# --------------------- EASYOCR SETUP (Keep as is) ---------------------
# reader = easyocr.Reader(['en'], gpu=False) # Assuming this is needed elsewhere

# --------------------- FILE LOADING (Keep as is) ---------------------
def load_route(path="jump_data.csv"):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Assuming JSON structure might be different, adjust if needed
        # This example assumes a list of dicts under a "jumps" key
        return data.get("jumps", [])
    elif ext == ".csv":
        # Using pandas now in main() for CSV loading and filtering
        # This function is primarily for the JSON case or as a fallback
        try:
            df = pd.read_csv(path)
            return df.to_dict(orient="records")
        except Exception as e:
            print(f"Error reading CSV with pandas: {e}")
            # Fallback basic CSV reading if needed
            jumps = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader_csv = csv.DictReader(f)
                    for row in reader_csv:
                        jumps.append(row)
                return jumps
            except Exception as e_basic:
                 print(f"Error with basic CSV reading: {e_basic}")
                 raise ValueError("Could not read CSV file.")

    else:
        raise ValueError("Unsupported file format. Please use .json or .csv")

# --------------------- GALAXY MAP CONTROL (Keep as is) ---------------------
def open_galaxy_map_and_plot_route():
    """
    Press F11 to trigger your AHK hotkey (assuming AHK is running).
    Wait a bit so Elite's galaxy map can open.
    """
    print("Triggering F11 (AHK Hotkey)...") # Added print for debugging
    pyautogui.press('f11')
    # Increased sleep time slightly, adjust as needed for your system/game load times
    time.sleep(18)
    print("Assumed Galaxy Map is ready.")
    return

# --------------------- GUI CLASS (Revised) ---------------------
class NeutronJumpGUI:
    def __init__(self, master, route_data):
        """
        Build the GUI. Also skip the first system (index 0) in the route_data,
        because that's the one you're already in and don't need to plot.
        """
        # --- Data Preparation ---
        if len(route_data) > 0:
            # Check if the first row seems like a header/current location based on typical CSV output
            # Example: If "Distance Remaining" is max or "Jumps" is 0. Adapt if needed.
            first_row = route_data[0]
            # A simple heuristic: if distance remaining equals distance to arrival, it might be the start
            # Or if jumps == 0. Adjust this logic based on your exact file format.
            # if first_row.get("Distance Remaining") == first_row.get("Distance To Arrival") or float(first_row.get("Jumps", 1)) == 0:
            #    print("Skipping first row (assumed current location).")
            #    route_data = route_data[1:]
            # Simpler: Always skip first row as per original logic
            print("Skipping first row (assumed current location).")
            route_data = route_data[1:]
        else:
             messagebox.showwarning("Empty Route", "The loaded route data is empty after skipping the first row.")
             # Decide how to handle this - perhaps disable button?
             # For now, allows GUI to load empty.

        self.master = master
        self.route_data = route_data
        self.total_jumps = len(route_data)
        self.current_jump = 0 # Start plotting from the first entry in the *filtered* list

        # --- Basic Window Setup ---
        self.master.title("Neutron Jump Overlay")
        # Increased width, adjusted height
        self.master.geometry("750x450+10+10")
        self.master.attributes("-topmost", True)
        # Optional: Add an icon (replace 'path/to/icon.ico' or .png)
        # try:
        #     # For Windows (.ico)
        #     # self.master.iconbitmap('path/to/icon.ico')
        #     # For cross-platform (.png)
        #     # icon_img = tk.PhotoImage(file='path/to/icon.png')
        #     # self.master.iconphoto(True, icon_img)
        # except tk.TclError:
        #     print("Icon file not found or invalid format.")


        # --- Styling ---
        self.style = ttk.Style()
        self.style.theme_use('clam') # 'clam', 'alt', 'default', 'classic' - experiment!

        # Define colors
        BG_COLOR = '#2E2E2E' # Dark grey
        FG_COLOR = '#E0E0E0' # Light grey
        HL_COLOR = '#FFFF99' # Pale Yellow highlight
        BTN_BG = '#3E8E41' # Darker Green
        BTN_FG = '#FFFFFF' # White
        DONE_BG = '#555555' # Grey for completed rows
        REATT_BG = '#FFA500' # Orange for reattempted rows

        # Configure root window background
        self.master.configure(bg=BG_COLOR)

        # Configure styles
        self.style.configure('.', background=BG_COLOR, foreground=FG_COLOR, font=('Segoe UI', 10))
        self.style.configure('TLabel', background=BG_COLOR, foreground=FG_COLOR, padding=(5, 5))
        self.style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=FG_COLOR)
        self.style.configure('Progress.TLabel', font=('Segoe UI', 12, 'bold'), foreground=FG_COLOR)
        self.style.configure('Status.TLabel', font=('Segoe UI', 11, 'bold'), foreground=HL_COLOR) # Yellow highlight color
        self.style.configure('TButton', background=BTN_BG, foreground=BTN_FG, font=('Segoe UI', 11, 'bold'), padding=(10, 5))
        self.style.map('TButton', background=[('active', '#4CAF50')]) # Slightly lighter green on hover/press

        # Treeview Style
        self.style.configure('Treeview', rowheight=25, fieldbackground=BG_COLOR, background=BG_COLOR, foreground=FG_COLOR)
        self.style.configure('Treeview.Heading', font=('Segoe UI', 10, 'bold'), background='#444444', foreground=FG_COLOR, padding=(5,5))
        self.style.map('Treeview.Heading', background=[('active', '#555555')])
        # Row tag styles
        self.style.configure("done.Treeview", background=DONE_BG) # Match Treeview item tag
        self.style.configure("reattempt.Treeview", background=REATT_BG) # Match Treeview item tag


        # --- Widgets ---
        self.title_label = ttk.Label(master, text="Neutron Jump Route", style='Title.TLabel')
        # Added padx for spacing from window edges
        self.title_label.pack(pady=(10, 5), padx=10)

        self.progress_label = ttk.Label(master, text="Progress: 0 / 0", style='Progress.TLabel')
        self.progress_label.pack(pady=5, padx=10)

        self.current_system_label = ttk.Label(master, text="Next jump instructions will appear here.",
                                             style='Status.TLabel',
                                             wraplength=700, # Adjust based on window width (750 - padding)
                                             justify=tk.LEFT) # Justify text left
        # Use fill='x' to allow wrapping to work correctly within packed width
        self.current_system_label.pack(pady=5, padx=10, fill='x')

        # Determine the columns from the route data; add a 'Status' column
        if self.total_jumps > 0:
            # Ensure standard columns exist if possible, handle missing keys gracefully
            base_cols = ["System Name", "Distance To Arrival", "Distance Remaining", "Neutron Star", "Refuel", "Jumps"]
            self.columns = [col for col in base_cols if col in route_data[0]]
            # Add any other columns present in the data that aren't standard
            self.columns.extend([col for col in route_data[0].keys() if col not in self.columns])
        else:
            self.columns = ["System Name", "Status"] # Default for empty data

        if "Status" not in self.columns:
            self.columns.append("Status")


        # --- Treeview Setup ---
        # Frame to hold Treeview and Scrollbar(s)
        tree_frame = ttk.Frame(master, style='TFrame') # Use TFrame for styling consistency
        tree_frame.pack(padx=10, pady=5, fill='both', expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=self.columns, show='headings', style='Treeview')

        # Define column widths and alignment (adjust these based on your data!)
        col_widths = {
            "System Name": 200,
            "Distance To Arrival": 100,
            "Distance Remaining": 100,
            "Neutron Star": 70,
            "Refuel": 70,
            "Jumps": 50,
            "Status": 150 # Give status column reasonable width
        }
        default_width = 100
        min_width = 40

        for col in self.columns:
            width = col_widths.get(col, default_width) # Get predefined width or use default
            self.tree.heading(col, text=col)
            # Use minwidth to prevent columns becoming too small
            self.tree.column(col, width=width, minwidth=min_width, anchor='w', stretch=True) # Anchor west (left), allow stretch

        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        # Optional Horizontal Scrollbar (uncomment if needed)
        # hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        # self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree.configure(yscrollcommand=vsb.set) # If only using vertical

        # Grid layout within the frame for Treeview and Scrollbar(s)
        self.tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        # Uncomment if using horizontal scrollbar
        # hsb.grid(row=1, column=0, sticky='ew')

        # Configure the frame's grid behavior
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)


        # --- Populate Treeview ---
        def format_value(val):
            # Attempt to format numbers, return original value otherwise
            try:
                num = float(val)
                # Format based on value - integers don't need decimal places
                if num == int(num):
                    return f"{int(num)}"
                else:
                    return f"{num:.1f}" # One decimal place for floats
            except (ValueError, TypeError):
                return str(val) # Ensure it's a string

        # Populate table – default "Status" to "Pending"
        for i, row_data in enumerate(self.route_data):
            # Ensure all expected columns have a value, defaulting to "N/A" or "Pending"
            row_values = []
            for col in self.columns:
                 if col == "Status":
                     val = row_data.get(col, "Pending")
                 else:
                     val = row_data.get(col, "N/A") # Use N/A for missing data
                 row_values.append(format_value(val))

            item_id = self.tree.insert("", "end", values=row_values, iid=str(i)) # Assign explicit iid

        # Configure row tags for background colors (must match style config names)
        self.tree.tag_configure("done", background=self.style.lookup('done.Treeview', 'background'))
        self.tree.tag_configure("reattempt", background=self.style.lookup('reattempt.Treeview', 'background'))


        # --- Button ---
        self.button = ttk.Button(master, text="Plot Next Jump (F10)",
                                command=self.next_jump, style='TButton')
        self.button.pack(pady=(5, 10)) # Slightly less padding above, more below

        # --- Hotkeys ---
        # Run listener in a thread so it doesn't block the GUI
        threading.Thread(target=self.hotkey_listener, daemon=True).start()
        self.update_progress_label() # Initial progress update
        # Set initial instruction text
        self.update_instruction_label()


    def hotkey_listener(self):
        """ Bind F10 to trigger next jump, Shift+F10 to reattempt. """
        # Using keyboard library requires running script as admin on Windows sometimes
        try:
            keyboard.add_hotkey("f10", self.queue_gui_update, args=(self.next_jump,))
            keyboard.add_hotkey("shift+f10", self.queue_gui_update, args=(self.reattempt_jump,))
            print("Hotkeys F10 (Next) and Shift+F10 (Reattempt) registered.")
            keyboard.wait() # Blocks this thread waiting for hotkeys
        except Exception as e:
             print(f"Error registering hotkeys (try running as admin?): {e}")
             messagebox.showerror("Hotkey Error", f"Could not register hotkeys (F10, Shift+F10).\nError: {e}\n\nTry running the script as Administrator.")


    def queue_gui_update(self, func_to_run):
        """ Safely schedule GUI updates from the keyboard listener thread. """
        # Tkinter GUI updates should happen in the main thread.
        # `after(0, ...)` schedules the function to run ASAP in the main event loop.
        self.master.after(0, func_to_run)

    def update_instruction_label(self):
        """ Updates the main instruction label based on the current state. """
        if self.current_jump >= self.total_jumps:
            self.current_system_label.config(text="Route Complete!")
        elif self.current_jump < len(self.route_data):
            current_row = self.route_data[self.current_jump]
            current_system_name = current_row.get("System Name") or list(current_row.values())[0]

            next_system_name = "Last jump!"
            if self.current_jump + 1 < self.total_jumps:
                 next_row = self.route_data[self.current_jump + 1]
                 next_system_name = next_row.get("System Name") or list(next_row.values())[0]

            # Use 'current_system_name' for the system to plot NOW.
            self.current_system_label.config(
                 text=f"Ready to plot: {current_system_name}\nPress F10 when neutron charged. Next system after this: {next_system_name}"
             )
        else:
             self.current_system_label.config(text="Waiting to start...")


    def next_jump(self):
        """ Handles the logic for plotting the next jump. """
        if self.current_jump >= self.total_jumps:
            messagebox.showinfo("Route Complete", "You have completed all planned jumps!")
            return

        # Get the row data for the *upcoming* jump
        row_data = self.route_data[self.current_jump]
        system_name = row_data.get("System Name", None)
        if not system_name: # Fallback if 'System Name' column doesn't exist or is empty
             try:
                 system_name = list(row_data.values())[0] # Use first column's value
             except IndexError:
                 messagebox.showerror("Error", f"Cannot find system name for jump {self.current_jump + 1}")
                 return

        # Copy system name BEFORE updating label (in case plotting takes time)
        try:
            pyperclip.copy(system_name)
            print(f"Copied '{system_name}' to clipboard.")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
            messagebox.showwarning("Clipboard Error", f"Could not copy '{system_name}' to clipboard.\nError: {e}\n\nYou may need to paste manually.")


        # Update GUI label immediately to show intent
        self.current_system_label.config(text=f"Plotting route to: {system_name}...")
        self.master.update_idletasks() # Force GUI update


        # Run the jump routine in a separate thread to avoid freezing GUI
        # Pass the specific system name and the index being processed
        threading.Thread(target=self.execute_jump, args=(system_name, self.current_jump), daemon=True).start()


    def execute_jump(self, system_name, jump_index_being_processed):
        """
        Triggers plotting, waits, then updates GUI. Runs in a separate thread.
        """
        try:
            open_galaxy_map_and_plot_route() # This includes the long sleep

            # --- Schedule GUI updates back in the main thread ---
            self.master.after(0, self.update_gui_after_jump, system_name, jump_index_being_processed)

        except Exception as e:
            print(f"Error during galaxy map automation: {e}")
            # Schedule error message display in main thread
            self.master.after(0, messagebox.showerror, "Automation Error", f"An error occurred during plotting:\n{e}")
            # Schedule resetting the label in main thread
            self.master.after(0, self.update_instruction_label)


    def update_gui_after_jump(self, plotted_system_name, completed_jump_index):
         """ Updates the Treeview and labels after a jump is plotted. Runs in main thread. """

         # Update the tree row's status using the *explicit iid* we set
         try:
             row_id = str(completed_jump_index) # Use the index passed to execute_jump
             if self.tree.exists(row_id):
                 new_status = f"Plotted route" # Keep status concise
                 self.tree.set(row_id, "Status", new_status)
                 # Remove previous tags and add 'done' tag
                 self.tree.item(row_id, tags=("done",))
                 # Ensure the completed row is visible
                 self.tree.see(row_id)
             else:
                  print(f"Warning: Tree item with ID '{row_id}' not found for update.")

         except Exception as e:
              print(f"Error updating treeview for index {completed_jump_index}: {e}")


         # --- IMPORTANT: Increment jump index AFTER successful plotting ---
         # Check if this update corresponds to the *current* jump index before incrementing
         # This prevents issues if jumps complete out of order or are reattempted
         if completed_jump_index == self.current_jump:
              self.current_jump += 1
              self.update_progress_label() # Update progress X / Y

         # Update instruction label for the *next* jump
         self.update_instruction_label()


    def reattempt_jump(self):
        """ Reattempt plotting the current jump without advancing. """
        if self.current_jump >= self.total_jumps:
            messagebox.showinfo("Route Complete", "Cannot reattempt, route finished.")
            return

        # Get the current row data (same as in next_jump)
        row_data = self.route_data[self.current_jump]
        system_name = row_data.get("System Name", None)
        if not system_name:
             try:
                 system_name = list(row_data.values())[0]
             except IndexError:
                 messagebox.showerror("Error", f"Cannot find system name for jump {self.current_jump + 1}")
                 return

        # Copy system name
        try:
            pyperclip.copy(system_name)
            print(f"Copied '{system_name}' for reattempt.")
        except Exception as e:
            print(f"Error copying to clipboard: {e}")
            messagebox.showwarning("Clipboard Error", f"Could not copy '{system_name}' to clipboard.\nError: {e}")

        # Update label to indicate reattempt
        self.current_system_label.config(text=f"Reattempting plot for: {system_name}...")
        self.master.update_idletasks()

        # Run plotting in a separate thread, passing the current index
        threading.Thread(target=self.execute_reattempt, args=(system_name, self.current_jump), daemon=True).start()


    def execute_reattempt(self, system_name, jump_index_being_reattempted):
        """ Triggers plotting for reattempt, waits, updates GUI. Runs in thread. """
        try:
            open_galaxy_map_and_plot_route() # This includes the long sleep

            # --- Schedule GUI updates back in the main thread ---
            self.master.after(0, self.update_gui_after_reattempt, system_name, jump_index_being_reattempted)

        except Exception as e:
            print(f"Error during galaxy map automation (reattempt): {e}")
            self.master.after(0, messagebox.showerror, "Automation Error", f"An error occurred during reattempt plotting:\n{e}")
            self.master.after(0, self.update_instruction_label) # Reset label


    def update_gui_after_reattempt(self, reattempted_system_name, reattempted_jump_index):
        """ Updates Treeview status for reattempt. Runs in main thread. """
        try:
            row_id = str(reattempted_jump_index)
            if self.tree.exists(row_id):
                new_status = f"Reattempted plot" # Concise status
                self.tree.set(row_id, "Status", new_status)
                self.tree.item(row_id, tags=("reattempt",)) # Apply reattempt style
                self.tree.see(row_id) # Ensure visible
            else:
                 print(f"Warning: Tree item with ID '{row_id}' not found for reattempt update.")
        except Exception as e:
             print(f"Error updating treeview for reattempt index {reattempted_jump_index}: {e}")

        # Update instruction label (shows the same 'next' jump)
        self.update_instruction_label()
        # Do NOT increment self.current_jump here


    def update_progress_label(self):
        """ Updates the 'Progress: X / Y' label. """
        self.progress_label.config(text=f"Progress: {self.current_jump} / {self.total_jumps}")


# --------------------- ENTRYPOINT (Revised for Pandas CSV loading) ---------------------
def main():
    # Create Tk instance early for dialogs
    root_dialog = tk.Tk()
    root_dialog.withdraw() # Hide the main window until needed

    file_path = filedialog.askopenfilename(
        parent=root_dialog, # Associate dialog with hidden root
        title="Select Neutron Jump Data File",
        filetypes=(("CSV files", "*.csv"), ("JSON files", "*.json"), ("All Files", "*.*"))
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected!", parent=root_dialog)
        root_dialog.destroy()
        return

    route_data = []
    try:
        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)
            # Clean column names (remove leading/trailing spaces)
            df.columns = df.columns.str.strip()

            # Check if necessary columns exist for filtering
            required_cols = ["Refuel", "Neutron Star"]
            if all(col in df.columns for col in required_cols):
                 # Ask about filtering normal jumps
                 route_includes_normal = messagebox.askyesno(
                     "Filter Jumps?",
                     "Does your route potentially include normal jumps (non-neutron, non-refuel)?\n\n"
                     "YES: Keep ONLY Neutron jumps OR Refuel stops.\n"
                     "NO:  Keep ALL jumps listed in the file.",
                     parent=root_dialog # Associate dialog
                 )

                 if route_includes_normal:
                     # Keep rows where EITHER Neutron Star is 'Yes' OR Refuel is 'Yes'
                     # Case-insensitive comparison
                     df_filtered = df[
                         (df["Neutron Star"].astype(str).str.lower() == "yes") |
                         (df["Refuel"].astype(str).str.lower() == "yes")
                     ]
                     print(f"Filtered CSV: Kept {len(df_filtered)} neutron/refuel jumps out of {len(df)} total.")
                     df = df_filtered # Use the filtered dataframe
                 else:
                      print("Using all jumps from CSV file.")

            else:
                print("Columns 'Refuel' or 'Neutron Star' not found in CSV, skipping filtering.")
                messagebox.showwarning("Filtering Skipped", "Could not find 'Refuel' and 'Neutron Star' columns in the CSV.\nWill use all rows.", parent=root_dialog)

            route_data = df.to_dict(orient="records")

        elif file_path.lower().endswith(".json"):
            route_data = load_route(file_path) # Assumes load_route handles JSON structure
            print(f"Loaded {len(route_data)} jumps from JSON.")

        else:
            messagebox.showerror("Error", "Unsupported file type selected (.csv or .json only).", parent=root_dialog)
            root_dialog.destroy()
            return

    except Exception as e:
        messagebox.showerror("File Load Error", f"Failed to load or process the file:\n{e}", parent=root_dialog)
        root_dialog.destroy()
        return

    # Destroy the temporary dialog root ONLY if successful so far
    root_dialog.destroy()

    # Launch the main GUI
    main_root = tk.Tk()
    gui = NeutronJumpGUI(main_root, route_data)
    main_root.mainloop()


if __name__ == "__main__":
    # Optional: Add error handling for imports if libraries might be missing
    try:
        import pandas # Check critical imports
        import keyboard
        import pyperclip
        import pyautogui
    except ImportError as err:
         # Use tkinter for error message if available
         try:
              root = tk.Tk()
              root.withdraw()
              messagebox.showerror("Missing Library", f"Required Python library not found: {err.name}\n\nPlease install it (e.g., using 'pip install {err.name}') and try again.")
              root.destroy()
         except tk.TclError: # Fallback if tkinter itself failed
              print(f"ERROR: Required Python library not found: {err.name}")
              print(f"Please install it (e.g., using 'pip install {err.name}') and try again.")
         exit() # Exit if libs missing

    main()