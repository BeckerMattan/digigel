import sys

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import (
    QShortcut, QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QCheckBox, QListWidget, QLineEdit, QFileDialog, QSpinBox, QMessageBox, QComboBox, QDoubleSpinBox,
    QGridLayout, QRadioButton, QMessageBox, QSizePolicy, QDesktopWidget, QStyleFactory
)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QIcon, QKeyEvent, QFont, QMovie
from PyQt5.QtGui import QPalette, QColor

import csv
import qdarkstyle
import random
import re
import matplotlib
from matplotlib.font_manager import fontManager
from traits.trait_types import false

matplotlib.use("Qt5Agg")  # Force Matplotlib to use PyQt5
#matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WXAgg', etc.

import matplotlib.pyplot as plt  # Explicit import of matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from form import Ui_MainWindow

import numpy as np
from numexpr.expressions import double
import os
import pandas as pd

import yaml

import hyperspy.api as hs  # Ensure Hyperspy is installed

hyperspy_path = os.path.dirname(hs.__file__)
os.environ["HYPERSPY_EXTENSIONS_PATH"] = os.path.join(hyperspy_path, "extensions")

import ruptures as rpt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import fsolve
import json
from scipy.optimize import curve_fit
from scipy.special import gamma, digamma
from scipy.stats import linregress
import pickle
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import time
from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

#from tslearn.clustering import TimeSeriesKMeans




def get_even_column_extrema(matrix_grid):
    """
    Given a 2D list of 2D NumPy arrays, returns the overall max and min values
    from the even-indexed columns (i.e., indices 0, 2, 4, ...) across all arrays.

    Parameters:
        matrix_grid (list of list of np.ndarray): 2D list of 2D numpy arrays.

    Returns:
        (float, float): Tuple of (max_value, min_value) from even columns.
    """
    even_columns_values = []

    for row in matrix_grid:
        for mat in row:
            mat = np.array(mat)
            if isinstance(mat, np.ndarray) and mat.ndim == 2:
                even_cols = mat[:, ::2]  # even-indexed columns
                even_columns_values.append(even_cols.flatten())
            else:
                raise ValueError("Each element must be a 2D NumPy array.")

    if not even_columns_values:
        raise ValueError("No valid data found in even-indexed columns.")

    all_values = np.concatenate(even_columns_values)
    return np.max(all_values), np.min(all_values)

def apply_dark_mode_to_canvas(canvas):
    """Apply dark mode styling to a Matplotlib canvas."""
    fig = canvas.figure
    ax_list = fig.axes

    # Set global figure background
    fig.patch.set_facecolor('#2e2e2e')  # dark gray
    fig.patch.set_edgecolor('#2e2e2e')

    for ax in ax_list:
        # Axes background
        ax.set_facecolor('#2e2e2e')

        # Axis labels, tick labels, and title colors
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color('white')

        # Grid (if needed)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

        # Legend (if present)
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor('#2e2e2e')
            frame.set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')



def apply_light_mode_to_canvas(canvas):
    """Revert a Matplotlib canvas to default (light) styling."""
    fig = canvas.figure
    ax_list = fig.axes

    # Set global figure background
    fig.patch.set_facecolor('white')
    fig.patch.set_edgecolor('white')

    for ax in ax_list:
        # Axes background
        ax.set_facecolor('white')

        # Axis labels, tick labels, and title colors
        ax.title.set_color('black')
        ax.xaxis.label.set_color('black')
        ax.yaxis.label.set_color('black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color('black')

        # Grid (optional: light gray)
        ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)

        # Legend (if present)
        legend = ax.get_legend()
        if legend:
            frame = legend.get_frame()
            frame.set_facecolor('white')
            frame.set_edgecolor('black')
            for text in legend.get_texts():
                text.set_color('black')





def my_exception_hook(exctype, value, tb):
    """
    Custom exception hook that shows a QMessageBox with a custom message when an unhandled
    exception occurs.
    """
    error_message = f"An unexpected error occurred:\n{value}"
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle("Application Error")
    msg.setText("The application has crashed, check your data or contact Mattan at Beckerm@post.bgu.ac.il.")
    msg.setInformativeText(error_message)
    font = QFont()
    font.setPointSize(14)  # adjust as needed
    msg.setFont(font)

    msg.exec_()

    # Optionally, log the exception details here if needed
    sys.exit(1)


def find_ipt_static(force_array, model="l2"):
    data = gaussian_filter1d(force_array, sigma=2)
    first_derivative = np.gradient(data)
    second_derivative = np.gradient(first_derivative)
    third_derivative = np.gradient(second_derivative)
    third_derivative = third_derivative[:-50]

    try:
        algo = rpt.Binseg(model=model).fit(first_derivative)
        change_points = algo.predict(n_bkps=4)
        ipt = change_points[0] - 1
        if ipt >= len(data):
            return len(data) - 1
        return ipt
    except Exception as e:
        print(f"find_ipt_static failed: {e}")
        return np.argmax(third_derivative)


def calculate_d0_for_trace(args):
    i, data, model = args
    sep_col = data[:, i * 2]
    force_col = data[:, i * 2 + 1]
    sep = sep_col[~np.isnan(sep_col)]
    force = force_col[~np.isnan(force_col)]
    if len(sep) == 0 or len(force) == 0:
        return i, 0, 0, 1e-6
    ipt = find_ipt_static(force, model)
    d0 = sep[ipt] - sep[-1]
    if d0 <= 0:
        error_message = f"An unexpected error occurred:\n{value}"
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Calculation Error")
        msg.setText("A trace yielded d0<=0, so the software cannot continue. Check your data.")
        msg.setInformativeText(error_message)
        font = QFont()
        font.setPointSize(14)  # adjust as needed
        msg.setFont(font)

        msg.exec_()
        return i, 0, 0, 0
    return i, ipt, ipt, d0


def convert_to_serializable(obj):
    """Recursively converts NumPy arrays, scalars, ranges, and None values to JSON-compatible formats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert ndarray to list
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return float(obj)  # Convert NumPy scalars to Python float
    elif isinstance(obj, range):
        return list(obj)  # Convert range to list
    elif obj is None:
        return "None"  # Convert None to a string for JSON compatibility
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}  # Convert nested dicts
    elif isinstance(obj, list):
        return [convert_to_serializable(value) for value in obj]  # Convert lists of objects
    else:
        return obj  # Return other types unchanged
def adapt_to_controller_settle_time(matrix:np.ndarray, time = 0):
    if time == 0 or not matrix.ndim == 2:
        return matrix
    total_time = 0
    for i in range(0, matrix.shape[1], 2):
        time_column = matrix[:, i]
        current_time = time_column[-1]-time_column[0]
        if i==0:
            total_time = current_time
        elif current_time > (total_time+1/1000) or current_time < (total_time-1/1000):
            raise ValueError("Not all relaxation traces are of the same time.\n Check your data.")
    frequency = total_time / matrix.shape[0]
    rows_to_remove = int(time/frequency)
    matrix = \
        matrix[rows_to_remove:, :]
    return matrix
"""
def convert_to_serializable_non_recursive(root):
    
    #Iteratively converts NumPy arrays, NumPy scalars, ranges, and None values
    #(as well as nested dicts and lists) to JSON-compatible formats.
    

    def basic_conversion(x):
        if isinstance(x, np.ndarray):
            return x.tolist()  # Convert ndarray to list
        elif isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
            return float(x)  # Convert NumPy scalars to Python float
        elif isinstance(x, range):
            return list(x)  # Convert range to list
        elif x is None:
            return "None"  # Convert None to a string
        elif isinstance(x, dict):
            return {}  # Create an empty dict to be filled later
        elif isinstance(x, list):
            return []  # Create an empty list to be filled later
        else:
            return x  # Other types remain unchanged

    # Convert the root element (might be a non-container)
    converted_root = basic_conversion(root)
    if not isinstance(root, (dict, list)):
        return converted_root

    # Each stack element is a tuple:
    # (original_container, converted_container, iterator, container_type)
    # For dicts, we iterate over .items(); for lists, we use enumerate.
    stack = []
    if isinstance(root, dict):
        stack.append((root, converted_root, iter(root.items()), 'dict'))
    else:  # it's a list
        stack.append((root, converted_root, iter(enumerate(root)), 'list'))

    while stack:
        orig, conv, it, cont_type = stack[-1]
        try:
            if cont_type == 'dict':
                key, value = next(it)
            else:  # list
                index, value = next(it)
        except StopIteration:
            stack.pop()
            continue

        # Convert the current value
        conv_value = basic_conversion(value)
        if cont_type == 'dict':
            conv[key] = conv_value
        else:
            conv.append(conv_value)

        # If the original value is itself a container, add it to the stack
        if isinstance(value, dict):
            stack.append((value, conv_value, iter(value.items()), 'dict'))
        elif isinstance(value, list):
            stack.append((value, conv_value, iter(enumerate(value)), 'list'))

    return converted_root
"""

def handle_design_upon_startup(self):
    self.setStyleSheet("""
        * {
            font-size: 14pt;
        }
    """)
    self.Header_label.setStyleSheet("""
        QLabel {
            font-size: 36pt;
            qproperty-alignment: 'AlignCenter';

        }
    """)
    self.DigiGel.setStyleSheet("""
        QLabel {
            font-size: 48pt;
            font-weight: bold;
            qproperty-alignment: 'AlignCenter';

        }
    """)
    self.copyright_label.setStyleSheet("""
        QLabel {
            font-size: 12pt;

        }
    """)
    self.handle_screen_change()
    screen = QApplication.primaryScreen()
    logical_geom = screen.availableGeometry()
    width = int(logical_geom.width() * 0.5)
    height = int(logical_geom.height() * 0.75)
    self.resize(width, height)
    self.setMinimumSize(int(logical_geom.width() * 0.25), int(logical_geom.height() * 0.375))  # or an appropriate lower bound


def save_experiment_params_pickle(self, folder_path, filename="experiment_params.pkl"):
    params = {
        "curr_force_sep_data": self.curr_force_sep_data,
        "d0": self.d0,
        "all_d0_vectors": self.all_d0_vectors,
        "all_a_vectors": self.all_a_vectors,
        "d0_default": self.d0_default,
        "all_ipts": self.all_ipts,
        "def_all_ipts": self.def_all_ipts,
        "all_ipt_vectors": self.all_ipt_vectors,
        "all_d0_vectors_merged": self.all_d0_vectors_merged,
        "all_a_vectors_merged": self.all_a_vectors_merged,
        "all_ipt_vectors_merged": self.all_ipt_vectors_merged,
        "all_E_star_vectors_merged": self.all_E_star_vectors_merged,
        "forcenum": self.forcenum,
        "curr_force_i": self.curr_force_i,
        "curr_sample_i": self.curr_sample_i,
        "duplicatenum": self.duplicatenum,
        "R_micron": self.R_micron,
        "n_clusters": self.n_clusters,
        "final_trace_number": self.final_trace_number,
        "current_model_index": self.current_model_index,
        "current_model": self.current_model,
        "traceNames": self.traceNames,
        "graphtitle_arr": self.graphtitle_arr,
        "forcevalues_arr": self.forcevalues_arr,
        "all_force_sep_AI_indices": self.all_force_sep_AI_indices,
        "all_force_time_AI_indices": self.all_force_time_AI_indices,
        "all_force_sep_matrices_before_filter": self.all_force_sep_matrices_before_filter,
        "all_force_time_matrices_before_filter": self.all_force_time_matrices_before_filter,
        "all_index_vectors_after_filtering": self.all_index_vectors_after_filtering,
        "all_force_sep_matrices_after_filter": self.all_force_sep_matrices_after_filter,
        "all_force_time_matrices_after_filter": self.all_force_time_matrices_after_filter,
        "all_index_vectors_after_filtering_merged": self.all_index_vectors_after_filtering_merged,
        "all_force_sep_matrices_after_filter_merged": self.all_force_sep_matrices_after_filter_merged,
        "all_force_time_matrices_after_filter_merged": self.all_force_time_matrices_after_filter_merged,
        "trace_names_after_filtering_merged_sep": self.trace_names_after_filtering_merged_sep,
        "trace_names_after_filtering_merged_time": self.trace_names_after_filtering_merged_time,
        "current_force_sep_matrix": self.current_force_sep_matrix,
        "current_force_time_matrix": self.current_force_time_matrix,
        "norm_x_vectors_avg": self.norm_x_vectors_avg,
        "visco_x_vectors_avg": self.visco_x_vectors_avg,
        "poro_x_vectors_avg": self.poro_x_vectors_avg,
        "norm_y_vectors_avg": self.norm_y_vectors_avg,
        "visco_y_vectors_avg": self.visco_y_vectors_avg,
        "poro_y_vectors_avg": self.poro_y_vectors_avg,
        "norm_x_vectors_std": self.norm_x_vectors_std,
        "visco_x_vectors_std": self.visco_x_vectors_std,
        "poro_x_vectors_std": self.poro_x_vectors_std,
        "norm_y_vectors_std": self.norm_y_vectors_std,
        "visco_y_vectors_std": self.visco_y_vectors_std,
        "poro_y_vectors_std": self.poro_y_vectors_std,
        "norm_x_vectors_UB": self.norm_x_vectors_UB,
        "visco_x_vectors_UB": self.visco_x_vectors_UB,
        "poro_x_vectors_UB": self.poro_x_vectors_UB,
        "norm_y_vectors_UB": self.norm_y_vectors_UB,
        "visco_y_vectors_UB": self.visco_y_vectors_UB,
        "poro_y_vectors_UB": self.poro_y_vectors_UB,
        "norm_x_vectors_LB": self.norm_x_vectors_LB,
        "visco_x_vectors_LB": self.visco_x_vectors_LB,
        "poro_x_vectors_LB": self.poro_x_vectors_LB,
        "norm_y_vectors_LB": self.norm_y_vectors_LB,
        "visco_y_vectors_LB": self.visco_y_vectors_LB,
        "poro_y_vectors_LB": self.poro_y_vectors_LB,
        "norm_all_matrices": self.norm_all_matrices,
        "visco_all_matrices": self.visco_all_matrices,
        "poro_all_matrices": self.poro_all_matrices,
        "force_sep_uploaded_flag": self.force_sep_uploaded_flag,
        "force_time_uploaded_flag": self.force_time_uploaded_flag,
        # "current_force_signal":self.current_force_signal,
        "current_sep_columns": self.current_sep_columns,
        "current_force_columns": self.current_force_columns,
        "debug": self.debug,
        "current_list_of_list_of_clusters": self.current_list_of_list_of_clusters,
        "current_index_vector": self.current_index_vector,
        "F_power": self.F_power,
        "R_power": self.R_power,
        "F_power_name": self.F_power_name,
        "R_power_name": self.R_power_name,
        "d0_averages_divided": self.d0_averages_divided,
        "d0_averages_final": self.d0_averages_final,
        "d0_stds_final": self.d0_stds_final,
        "current_force_sep_CSV_length": self.current_force_sep_CSV_length,
        "current_force_time_CSV_length": self.current_force_time_CSV_length,
        "debug_upload_csv_i": self.debug_upload_csv_i,
        "variance_visco_vector": self.variance_visco_vector,
        "variance_poro_vector": self.variance_poro_vector,
        "variance_poro_vectors": self.variance_poro_vectors,
        "experiment_name_DigiGel": self.experiment_name,
        "conical_indenter": self.conical_indenter,
        "spherical_indenter": self.spherical_indenter,
        "half_angle_indenter_rad": self.half_angle_indenter_rad,
        "tan_alpha_indenter": self.tan_alpha_indenter,
        "current_tab": self.current_tab,
        "D_p": self.D_p,
        "G": self.G,
        "nu": self.nu,
        "beta": self.beta,
        "C_beta": self.C_beta,
        "D_p_std": self.D_p_std,
        "G_std": self.G_std,
        "nu_std": self.nu_std,
        "beta_std": self.beta_std,
        "C_beta_std": self.C_beta_std,
        "D_p_tot": self.D_p_tot,
        "G_tot": self.G_tot,
        "nu_tot": self.nu_tot,
        "beta_tot": self.beta_tot,
        "C_beta_tot": self.C_beta_tot,
        "D_p_std_tot": self.D_p_std_tot,
        "G_std_tot": self.G_std_tot,
        "nu_std_tot": self.nu_std_tot,
        "beta_std_tot": self.beta_std_tot,
        "C_beta_std_tot": self.C_beta_std_tot,
        "r_squared_visco": self.r_squared_visco,
        "r_squared_poro": self.r_squared_poro,
        "fit_type": self.fit_type,
        "visco_y_vectors_avg_fits": self.visco_y_vectors_avg_fits,
        "poro_y_vectors_avg_fits": self.poro_y_vectors_avg_fits,
        "visco_y_vectors_UB_fits": self.visco_y_vectors_UB_fits,
        "poro_y_vectors_UB_fits": self.poro_y_vectors_UB_fits,
        "visco_y_vectors_LB_fits": self.visco_y_vectors_LB_fits,
        "poro_y_vectors_LB_fits": self.poro_y_vectors_LB_fits,
        "all_G_values": self.all_G_values,
        "all_nu_values": self.all_nu_values,
        "poro_fits_performed_already": self.poro_fits_performed_already,
        "visco_fits_performed_already": self.visco_fits_performed_already,
        "r_squared_visco_values": self.r_squared_visco_values,
        "r_squared_poro_values": self.r_squared_poro_values,
        "saving_version": self.saving_version,
        "all_headers_vectors_before_filter": self.all_headers_vectors_before_filter,
        "all_headers_vectors_taken_out": self.all_headers_vectors_taken_out,
        "filtered_out_indices": self.filtered_out_indices,
        "filtered_out_indices_reason": self.filtered_out_indices_reason,

    }

    filepath = os.path.join(folder_path, filename)
    with open(filepath, "wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Parameters saved (Pickle) to: {filepath}")

"""
def save_experiment_params(self, folder_path, filename="experiment_params.json"):
    #Saves experiment parameters as a JSON file (excluding PyQt objects).
    params = {
        "curr_force_sep_data": self.curr_force_sep_data,
        "d0": self.d0,
        "all_d0_vectors": self.all_d0_vectors,
        "all_a_vectors": self.all_a_vectors,
        "d0_default": self.d0_default,
        "all_ipts": self.all_ipts,
        "def_all_ipts": self.def_all_ipts,
        "all_ipt_vectors": self.all_ipt_vectors,
        "all_d0_vectors_merged": self.all_d0_vectors_merged,
        "all_a_vectors_merged": self.all_a_vectors_merged,
        "all_ipt_vectors_merged": self.all_ipt_vectors_merged,
        "all_E_star_vectors_merged": self.all_E_star_vectors_merged,
        "forcenum": self.forcenum,
        "curr_force_i": self.curr_force_i,
        "curr_sample_i": self.curr_sample_i,
        "duplicatenum": self.duplicatenum,
        "R_micron": self.R_micron,
        "n_clusters": self.n_clusters,
        "final_trace_number": self.final_trace_number,
        "current_model_index": self.current_model_index,
        "current_model": self.current_model,
        "traceNames": self.traceNames,
        "graphtitle_arr": self.graphtitle_arr,
        "forcevalues_arr": self.forcevalues_arr,
        "all_force_sep_AI_indices": self.all_force_sep_AI_indices,
        "all_force_time_AI_indices": self.all_force_time_AI_indices,
        "all_force_sep_matrices_before_filter": self.all_force_sep_matrices_before_filter,
        "all_force_time_matrices_before_filter": self.all_force_time_matrices_before_filter,
        "all_index_vectors_after_filtering": self.all_index_vectors_after_filtering,
        "all_force_sep_matrices_after_filter": self.all_force_sep_matrices_after_filter,
        "all_force_time_matrices_after_filter": self.all_force_time_matrices_after_filter,
        "all_index_vectors_after_filtering_merged": self.all_index_vectors_after_filtering_merged,
        "all_force_sep_matrices_after_filter_merged": self.all_force_sep_matrices_after_filter_merged,
        "all_force_time_matrices_after_filter_merged": self.all_force_time_matrices_after_filter_merged,
        "trace_names_after_filtering_merged_sep": self.trace_names_after_filtering_merged_sep,
        "trace_names_after_filtering_merged_time": self.trace_names_after_filtering_merged_time,
        "current_force_sep_matrix": self.current_force_sep_matrix,
        "current_force_time_matrix": self.current_force_time_matrix,
        "norm_x_vectors_avg": self.norm_x_vectors_avg,
        "visco_x_vectors_avg": self.visco_x_vectors_avg,
        "poro_x_vectors_avg": self.poro_x_vectors_avg,
        "norm_y_vectors_avg": self.norm_y_vectors_avg,
        "visco_y_vectors_avg": self.visco_y_vectors_avg,
        "poro_y_vectors_avg": self.poro_y_vectors_avg,
        "norm_x_vectors_std": self.norm_x_vectors_std,
        "visco_x_vectors_std": self.visco_x_vectors_std,
        "poro_x_vectors_std": self.poro_x_vectors_std,
        "norm_y_vectors_std": self.norm_y_vectors_std,
        "visco_y_vectors_std": self.visco_y_vectors_std,
        "poro_y_vectors_std": self.poro_y_vectors_std,
        "norm_x_vectors_UB": self.norm_x_vectors_UB,
        "visco_x_vectors_UB": self.visco_x_vectors_UB,
        "poro_x_vectors_UB": self.poro_x_vectors_UB,
        "norm_y_vectors_UB": self.norm_y_vectors_UB,
        "visco_y_vectors_UB": self.visco_y_vectors_UB,
        "poro_y_vectors_UB": self.poro_y_vectors_UB,
        "norm_x_vectors_LB": self.norm_x_vectors_LB,
        "visco_x_vectors_LB": self.visco_x_vectors_LB,
        "poro_x_vectors_LB": self.poro_x_vectors_LB,
        "norm_y_vectors_LB": self.norm_y_vectors_LB,
        "visco_y_vectors_LB": self.visco_y_vectors_LB,
        "poro_y_vectors_LB": self.poro_y_vectors_LB,
        "norm_all_matrices": self.norm_all_matrices,
        "visco_all_matrices": self.visco_all_matrices,
        "poro_all_matrices": self.poro_all_matrices,
        "force_sep_uploaded_flag": self.force_sep_uploaded_flag,
        "force_time_uploaded_flag": self.force_time_uploaded_flag,
        # "current_force_signal":self.current_force_signal,
        "current_sep_columns": self.current_sep_columns,
        "current_force_columns": self.current_force_columns,
        "debug": self.debug,
        "current_list_of_list_of_clusters": self.current_list_of_list_of_clusters,
        "current_index_vector": self.current_index_vector,
        "F_power": self.F_power,
        "R_power": self.R_power,
        "F_power_name": self.F_power_name,
        "R_power_name": self.R_power_name,
        "d0_averages_divided": self.d0_averages_divided,
        "d0_averages_final": self.d0_averages_final,
        "d0_stds_final": self.d0_stds_final,
        "current_force_sep_CSV_length": self.current_force_sep_CSV_length,
        "current_force_time_CSV_length": self.current_force_time_CSV_length,
        "debug_upload_csv_i": self.debug_upload_csv_i,
        "variance_visco_vector": self.variance_visco_vector,
        "variance_poro_vector": self.variance_poro_vector,
        "variance_poro_vectors": self.variance_poro_vectors,
        "experiment_name_DigiGel": self.experiment_name,
        "conical_indenter": self.conical_indenter,
        "spherical_indenter": self.spherical_indenter,
        "half_angle_indenter_rad": self.half_angle_indenter_rad,
        "tan_alpha_indenter": self.tan_alpha_indenter,
        "current_tab": self.current_tab,
        "D_p": self.D_p,
        "G": self.G,
        "nu": self.nu,
        "beta": self.beta,
        "C_beta": self.C_beta,
        "D_p_std": self.D_p_std,
        "G_std": self.G_std,
        "nu_std": self.nu_std,
        "beta_std": self.beta_std,
        "C_beta_std": self.C_beta_std,
        "D_p_tot": self.D_p_tot,
        "G_tot": self.G_tot,
        "nu_tot": self.nu_tot,
        "beta_tot": self.beta_tot,
        "C_beta_tot": self.C_beta_tot,
        "D_p_std_tot": self.D_p_std_tot,
        "G_std_tot": self.G_std_tot,
        "nu_std_tot": self.nu_std_tot,
        "beta_std_tot": self.beta_std_tot,
        "C_beta_std_tot": self.C_beta_std_tot,
        "r_squared_visco": self.r_squared_visco,
        "r_squared_poro": self.r_squared_poro,
        "fit_type": self.fit_type,
        "visco_y_vectors_avg_fits": self.visco_y_vectors_avg_fits,
        "poro_y_vectors_avg_fits": self.poro_y_vectors_avg_fits,
        "visco_y_vectors_UB_fits": self.visco_y_vectors_UB_fits,
        "poro_y_vectors_UB_fits": self.poro_y_vectors_UB_fits,
        "visco_y_vectors_LB_fits": self.visco_y_vectors_LB_fits,
        "poro_y_vectors_LB_fits": self.poro_y_vectors_LB_fits,
        "all_G_values": self.all_G_values,
        "all_nu_values": self.all_nu_values,
        "poro_fits_performed_already": self.poro_fits_performed_already,
        "visco_fits_performed_already": self.visco_fits_performed_already,
        "r_squared_visco_values": self.r_squared_visco_values,
        "r_squared_poro_values": self.r_squared_poro_values,
        "saving_version": self.saving_version,
        "all_headers_vectors_before_filter": self.all_headers_vectors_before_filter,
        "all_headers_vectors_taken_out": self.all_headers_vectors_taken_out,
        "filtered_out_indices": self.filtered_out_indices,
        "filtered_out_indices_reason": self.filtered_out_indices_reason,

    }

    params_serializable = convert_to_serializable(params)

    filepath = os.path.join(folder_path, filename)
    with open(filepath, "w") as f:
        json.dump(params_serializable, f, indent=4)
    print(f"Parameters saved to: {filepath}")
"""

def select_double_columns(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Selects specific columns and their next consecutive column from a 2D NumPy array.

    Parameters:
    - matrix (np.ndarray): The input 2D array.
    - indices (np.ndarray): A 1D array of column indices.

    Returns:
    - np.ndarray: A new 2D array containing the selected columns and their consecutive ones.
    """
    matrix = np.array(matrix)
    debug_indices = [[i, i + 1] for i in 2 * indices if i < matrix.shape[1] + 1]
    selected_columns = np.hstack([matrix[:, [i, i + 1]] for i in 2 * indices if i < matrix.shape[1] + 1])
    return selected_columns


def add_sheet_to_existing_excel(file_path, new_sheet_name, data, headers=None):
    """Adds a new sheet to an existing Excel file without overwriting it."""

    # Load existing workbook
    writer = pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="new")

    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Write to a new sheet
    df.to_excel(writer, index=False, sheet_name=new_sheet_name)

    # Save changes
    writer.close()
    print(f"Added new sheet '{new_sheet_name}' to {file_path}")


def save_ndarray_to_excel(array, folder_path, filename="output.xlsx", headers=None, sheet_name="Sheet1"):
    """Saves a NumPy ndarray as an Excel (.xlsx) file in a designated folder with column headers."""

    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create full file path
    file_path = os.path.join(folder_path, filename)

    # Convert ndarray to DataFrame with headers
    df = pd.DataFrame(array, columns=headers)

    # Save to Excel
    df.to_excel(file_path, index=False, sheet_name=sheet_name, engine="openpyxl")

    print(f"Saved ndarray to {file_path}")


def save_list_to_excel(string_list, folder_path, filename="output.xlsx", header=None, sheet_name="Sheet1"):
    """
    Saves a list of strings as an Excel (.xlsx) file in a designated folder with a column header.

    Parameters:
      string_list (list of str): The list of strings to be saved.
      folder_path (str): Path to the folder where the file will be saved.
      filename (str): The name of the Excel file (default "output.xlsx").
      header (str): Column header for the list. If None, a default header "Value" is used.
      sheet_name (str): The name of the Excel sheet (default "Sheet1").
    """
    if not string_list:
        return
    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create full file path
    file_path = os.path.join(folder_path, filename)

    # Set a default header if not provided
    if header is None:
        header = "Value"

    # Convert list of strings to a DataFrame with one column
    if header is not None:
        df = pd.DataFrame(string_list, columns=header)
    else:
        df = pd.DataFrame(string_list)

    # Save the DataFrame to Excel
    df.to_excel(file_path, index=False, sheet_name=sheet_name)

    print(f"Saved list to {file_path}")


def save_ndarray_to_csv(array, folder_path, filename="output.xlsx", headers=None, sheet_name="Sheet1"):
    """Saves a NumPy ndarray as an Excel (.xlsx) file in a designated folder with column headers."""

    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create full file path
    file_path = os.path.join(folder_path, filename)

    # Convert ndarray to DataFrame with headers
    df = pd.DataFrame(array, columns=headers)

    # Save to Excel
    df.to_csv(file_path, index=False)

    print(f"Saved ndarray to {file_path}")


def load_experiment_params(filepath):
    """Loads experiment parameters from a JSON file."""
    # with open(filepath, "r") as f:
    #    params = json.load(f)
    # return params
    with open(filepath, "rb") as f:
        params = pickle.load(f)
    return params


def find_maximum_length_column(input_list: list[list[float]]) -> int:
    transposed = list(map(list, zip(*input_list)))
    max_length = 0
    for row in transposed:
        if len(row) > max_length:
            max_length = len(row)

    return max_length


def fill_column(column, total_length):
    if isinstance(column, list):
        column = pd.Series(column)

    column_ffilled = column.ffill()  # Forward-fill missing values in the column
    if len(column_ffilled) < total_length:
        last_value = column_ffilled.iloc[-1]
        additional_values = pd.Series([last_value] * (total_length - len(column_ffilled)))
        column_ffilled = pd.concat([column_ffilled, additional_values], ignore_index=True)
    return column_ffilled


def find_ipt(self, data: list):
    # if self.debug:
    # return int(len(data)/2)
    data = gaussian_filter1d(data, sigma=2)
    first_derivative = np.gradient(data)
    second_derivative = np.gradient(first_derivative)
    third_derivative = np.gradient(second_derivative)
    third_derivative = third_derivative[:-50]
    max_curvature_index = np.argmax(third_derivative)
    self.current_model = self.model_find_d0_combobox.currentText()
    algo = rpt.Binseg(model=self.model_find_d0_combobox.currentText()).fit(
        first_derivative)  # Use Binary Segmentation with "l2" norm
    change_points = algo.predict(n_bkps=4)  # Specify one change point
    ipt = change_points[0] - 1
    if ipt >= len(data):
        return len(data) - 1
    return change_points[0] - 1  # Adjust to zero-based index, if needed

    # return max_curvature_index


def fill_empty_space_2D_list(input_list: list[list[float]], length) -> list[list[float]]:
    # Transpose the input list to access columns as rows
    input_list = pd.DataFrame(input_list)

    output_df = input_list.apply(lambda col: fill_column(col, length), axis=0)
    output_2d_list = output_df.values.tolist()

    return output_2d_list


def save_matrix_as_csv(matrix: list, name: str):
    with open(name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(matrix)


def finalize_column(column, total_length):
    column_ffilled = column.ffill()  # Forward-fill missing values
    column_ffilled = column_ffilled.bfill()  # Backward-fill if leading values are NaN
    if len(column_ffilled) < total_length:
        last_value = column_ffilled.iloc[-1]
        additional_values = pd.Series([last_value] * (total_length - len(column_ffilled)))
        column_ffilled = pd.concat([column_ffilled, additional_values], ignore_index=True)
    return column_ffilled


def calculate_length_scale(self, d0_value, R_value):
    # self.conical_indenter = False
    # self.spherical_indenter = True

    if self.conical_indenter:
        return (2 / np.pi) * d0_value * self.tan_alpha_indenter

    def Sneddon(a, d0=d0_value, R=R_value):
        return 0.5 * a * np.log((R + a) / (R - a)) - d0

    a = fsolve(Sneddon, d0_value, xtol=1e-8, maxfev=1000)
    return a[0]


def append_column(mother_list, daughter_list, index):
    """
    Appends the index-th column of mother_list to daughter_list. If daughter_list or its rows
    are not lists, they will be converted into lists containing the corresponding value from mother_list.

    Parameters:
        mother_list (list of lists): The 2D list from which the column is extracted.
        daughter_list (list of lists or int): The 2D list where the column will be appended.
        index (int): The column index to extract from mother_list.

    Returns:
        list of lists: The updated daughter_list with the new column added.
    """
    # Ensure daughter_list is a list
    if not isinstance(daughter_list, list):
        daughter_list = []
    # Ensure daughter_list has the same number of rows as mother_list
    while len(daughter_list) < len(mother_list):
        daughter_list.append([])

    for i in range(len(mother_list)):
        # Ensure valid indexing for mother_list
        if index < len(mother_list[i]):
            value_to_add = mother_list[i][index]
        else:
            print("len(mother_list)=", len(mother_list[i]))
            value_to_add = None  # Use None if the index is out of bounds

        # If the current row in daughter_list is not a list, initialize it as a list
        if not isinstance(daughter_list[i], list):
            daughter_list[i] = []

        # Append the value to the current row in daughter_list
        daughter_list[i].append(value_to_add)

    return daughter_list


def process_2d_list(input_list):
    # Convert the 2D list to a NumPy array for easier column extraction
    matrix = np.array(input_list)
    total_length = matrix.shape[0]

    for i in range(matrix.shape[1]):
        signal = matrix[:, i]
        signal = signal[~np.isnan(signal)]
        x = range(len(signal))
        y = np.array(signal)
        xvals = (1 / total_length * len(signal)) * np.array(range(total_length))
        processed_signal = np.interp(xvals, x, y)
        matrix[:, i] = processed_signal.tolist()

    # Transpose back to row-major order for the 2D list
    return matrix


def process_2d_list_specific_length(input_list, total_length):
    # Convert the 2D list to a NumPy array for easier column extraction
    matrix = np.array(input_list)
    # total_length = matrix.shape[0]
    new_matrix = np.zeros((total_length, input_list.shape[1]))

    for i in range(matrix.shape[1]):
        signal = matrix[:, i]
        signal = signal[~np.isnan(signal)]
        x = range(len(signal))
        y = np.array(signal)
        xvals = (1 / total_length * len(signal)) * np.array(range(total_length))
        processed_signal = np.interp(xvals, x, y)
        new_matrix[:, i] = processed_signal.tolist()

    # Transpose back to row-major order for the 2D list
    return new_matrix


def Normalize_norm_2d_list(input_list, force_value):
    matrix = np.array(input_list)

    for i in range(int(0.5 * matrix.shape[1])):
        force = matrix[:, 2 * i + 1]
        force = force[~np.isnan(force)]
        time = matrix[:, 2 * i]
        time = time[~np.isnan(time)]
        time = time - time[0]
        force = force - force[0] + force_value
        matrix[:, 2 * i + 1] = force
        matrix[:, 2 * i] = time
    return matrix

def Normalize_x_val_2d_forcetime_list(input_list):
    matrix = np.array(input_list)

    for i in range(int(0.5 * matrix.shape[1])):
        time = matrix[:, 2 * i]
        time = time[~np.isnan(time)]
        time = time - time[0]
        matrix[:, 2 * i] = time
    return matrix


def Normalize_force_2d_list(input_list):
    matrix = np.array(input_list)

    for i in range(int(0.5 * matrix.shape[1])):
        force = matrix[:, 2 * i + 1]
        force = force[~np.isnan(force)]
        f0 = force[0]
        f_end = force[-1]
        force = (force - f_end) / (f0 - f_end)
        matrix[:, 2 * i + 1] = force

    return matrix


def Normalize_time_2d_list(input_list, a_list):
    matrix = np.array(input_list)
    a_vector = np.array(a_list)

    for i in range(int(0.5 * matrix.shape[1])):
        time = matrix[:, 2 * i]
        time = time[~np.isnan(time)]
        a = a_vector[i]
        time = time / (a ** 2)
        matrix[:, 2 * i] = time
    return matrix


def Calculate_average_and_std(input_list):
    matrix = np.array(input_list)
    x_matrix = matrix[:, ::2]
    y_matrix = matrix[:, 1::2]
    x_averages = np.mean(x_matrix, axis=1)
    y_averages = np.mean(y_matrix, axis=1)
    x_stds = np.std(x_matrix, axis=1)
    y_stds = np.std(y_matrix, axis=1)

    return x_averages, y_averages, x_stds, y_stds


def normalize_forcetime_list(self, input_list):
    matrix = np.array(input_list)

    for i in range(int(0.5 * matrix.shape[1])):
        time = matrix[:, 2 * i]
        if time.size == 0:
            # Handle the empty array case appropriately
            # For example, return the empty array or raise a custom error
            print("Warning: Received an empty time array.")
        time = time[~np.isnan(np.array(time))]
        force = matrix[:, 2 * i + 1]
        force = force[~np.isnan(force)]
        if time[0]:
            time = time - time[0]
            force = force - force[0] + self.forcevalues_arr[self.curr_force_i] * 1e-9
            matrix[:, 2 * i] = time
            matrix[:, 2 * i + 1] = force
        else:
            print("Warning: Received an empty time array.")
    return matrix


def new_calculate_cluster_graph(data):
    sse = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        sse.append(kmeans.inertia_)
    return k_range, sse
"""
def new_calculate_cluster_graph_pca(data):
    # Step 1: Reduce dimensionality
    # n_components=0.99 means "keep enough components to explain 99% of the variance"
    # This usually reduces 1000 points down to ~5-10 meaningful features.
    pca = PCA(n_components=0.99)
    data_pca = pca.fit_transform(data)

    # Optional: Print how many components it kept
    print(f"PCA reduced data from {data.shape[1]} time points to {data_pca.shape[1]} components.")

    sse = []
    k_range = range(1, 11)

    # Step 2: Run K-Means on the REDUCED data
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data_pca)
        sse.append(kmeans.inertia_)

    return k_range, sse
"""
"""
def new_calculate_cluster_graph_ts(data):
    costs = []
    k_range = range(1, 11)

    for k in k_range:
        # Optimization 1: n_jobs=-1 uses all available CPU cores
        # Optimization 2: global_constraint="sakoe_chiba" limits the DTW search window
        # Optimization 3: sakoe_chiba_radius defines the window size (e.g., 10% of length)
        model = TimeSeriesKMeans(
            n_clusters=k,
            metric="dtw",
            n_init=2,
            max_iter=50,
            random_state=0,
            n_jobs=-1,
            metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 3}
        )
        model.fit(data)
        costs.append(model.inertia_)

    return k_range, costs
"""

def calculate_cluster_number(matrix: np.ndarray, matplotlib_element_for_elbow: QtWidgets.QWidget, is_forcetime: bool = False):
    # matrix = matrix.T
    sep_columns = matrix[:, 0::2]

    # sep_columns = np.array([matrix[i] for i in range(len(matrix)) if i % 2 == 0])  # Even-indexed elements
    force_columns = matrix[:, 1::2]

    # force_columns = np.array([matrix[i] for i in range(len(matrix)) if i % 2 != 0])  # Odd-indexed elements

    sep_columns = sep_columns.T
    force_columns = force_columns.T
    # force_columns = (force_columns.T)*(10**9)

    # sep_signal = hs.signals.Signal1D(sep_columns)
    #####force_signal = hs.signals.Signal1D(force_columns)
    # elbow
    # elbow_result = force_signal.estimate_number_of_clusters(cluster_source="signal", metric="elbow")

    # %matplotlib inline  - if you want to show the elbow graph
    # force_signal.plot_cluster_metric()
    # plt.show()
    # n_clusters = force_signal.learning_results.estimated_number_of_clusters
    cluster_numbers, cluster_metrics = new_calculate_cluster_graph(force_columns)
    n_clusters = local_claculate_cluster_number(cluster_numbers, cluster_metrics)
    if not isinstance(n_clusters, int):
        if (n_clusters is None):
            n_clusters = 3
        elif (isinstance(n_clusters, list)):
            n_clusters = n_clusters[-1]
        else:
            n = 3
    # if (n_clusters<3): n_clusters=3;
    # return n_clusters, force_signal, sep_columns, force_columns
    return n_clusters, sep_columns, force_columns


def local_claculate_cluster_number(cluster_numbers, cluster_metrics):
    slopes = np.diff(cluster_metrics)
    steepest_index = np.argmin(slopes)
    k = cluster_numbers[steepest_index + 1]
    best_k = max(min(cluster_numbers), min(k, max(cluster_numbers)))
    return best_k


def plot_elbow_in_ai_viewing_graph(self, force_columns):
    """
    Plot the elbow graph in the self.AI_Viewing_Graph Matplotlib canvas.
    """
    # Step 1: Ensure clustering results are available
    """if not hasattr(force_signal, "learning_results") or not force_signal.learning_results:
        raise ValueError("Clustering metrics not found. Ensure `estimate_number_of_clusters()` has been called.")"""

    # Step 2: Extract data from learning_results
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((16 / 1440) * height)
    title_size = int((32 / 1440) * height)
    cluster_numbers, cluster_metrics = new_calculate_cluster_graph(force_columns)

    # cluster_metrics = force_signal.learning_results.cluster_metric_data  # List of cluster numbers
    # cluster_numbers = np.array(force_signal.learning_results.cluster_metric_index)  # Corresponding metric values

    # Step 3: Plot the elbow graph on the AI_Viewing_Graph canvas
    self.AI_Viewing_Graph.ax.clear()  # Clear previous plots
    self.AI_Viewing_Graph.ax.plot(cluster_numbers, cluster_metrics, marker="o", linestyle="None", markersize=15)
    self.AI_Viewing_Graph.ax.set_title("Elbow Method for\nOptimal Group Number", fontsize=title_size)
    self.AI_Viewing_Graph.ax.set_xlabel("Number of Groups", fontsize=font_size)
    self.AI_Viewing_Graph.ax.set_ylabel("Deviation Metric (WCSS)", fontsize=font_size)
    self.AI_Viewing_Graph.ax.grid(True)
    self.AI_Viewing_Graph.fig.tight_layout()

    # Step 4: Refresh the canvas
    self.AI_Viewing_Graph.draw()


def plot_graph_in_ai_viewing_graph(self, x, y, isrelax: bool):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)
    rect = screen.geometry()
    logical_width = rect.width()
    logical_height = rect.height()
    dpr = screen.devicePixelRatio()
    actual_width = int(logical_width * dpr)
    actual_height = int(logical_height * dpr)

    self.AI_Viewing_Graph.ax.clear()  # Clear previous plots
    self.AI_Viewing_Graph.ax.plot(x, y, marker="None", linestyle="-", linewidth=3)
    if (isrelax):
        self.AI_Viewing_Graph.ax.set_title("Relaxation Curve", fontsize=title_size)
        self.AI_Viewing_Graph.ax.set_xlabel("time [s]", fontsize=font_size)
        self.AI_Viewing_Graph.ax.set_ylabel("force [N]", fontsize=font_size)
    else:
        self.AI_Viewing_Graph.ax.set_title("Indentation Curve", fontsize=title_size)
        self.AI_Viewing_Graph.ax.set_xlabel("separation [m]", fontsize=font_size)
        self.AI_Viewing_Graph.ax.set_ylabel("force [N]", fontsize=font_size)

    self.AI_Viewing_Graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.AI_Viewing_Graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.AI_Viewing_Graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.AI_Viewing_Graph.fig.tight_layout()
    self.AI_Viewing_Graph.ax.tick_params(axis='both', labelsize=font_size)
    if not isrelax:
        self.AI_Viewing_Graph.ax.set_xlim(self.min_sep_val, self.max_sep_val)

    self.AI_Viewing_Graph.ax.grid(True)
    self.AI_Viewing_Graph.draw()


def plot_graph_in_find_d0_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)

    self.Force_Sep_Graph.ax.clear()  # Clear previous plots
    self.curr_force_sep_data = np.array(self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i])
    d0_values = self.d0
    ipt_values = self.all_ipts

    i = self.force_curve_list.currentRow()
    force = self.curr_force_sep_data[:, i * 2 + 1]
    force = force[~np.isnan(force)]
    sep = self.curr_force_sep_data[:, i * 2]
    sep = sep[~np.isnan(sep)]
    if len(ipt_values)==0:
        ipt_values=[0]
    ipt = int(ipt_values[i])
    d0 = d0_values[i]
    length_unit_text = self.R_power_name
    self.Indentation_depth_label.setText(
        "Indentation Depth = " + str(round(d0 * 10 ** (-self.R_power), 2)) + " " + length_unit_text)

    sep2 = [sep[ipt], sep[-1], sep[-1]]
    force2 = [force[ipt], force[ipt], force[-1]]
    x = sep
    y = force

    self.Force_Sep_Graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3)
    self.Force_Sep_Graph.ax.plot(sep2, force2, marker="o", linestyle="--", linewidth=3)
    self.Force_Sep_Graph.ax.set_title("Indentation Curve", fontsize=title_size)
    self.Force_Sep_Graph.ax.set_xlabel("separation [m]", fontsize=font_size)
    self.Force_Sep_Graph.ax.set_ylabel("force [N]", fontsize=font_size)
    self.Force_Sep_Graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.Force_Sep_Graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.Force_Sep_Graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.Force_Sep_Graph.ax.tick_params(axis='both', labelsize=font_size)

    self.Force_Sep_Graph.ax.grid(True)
    self.Force_Sep_Graph.fig.tight_layout()
    self.Force_Sep_Graph.draw()


def plot_default_graph_in_find_d0_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)

    self.Force_Sep_Graph.ax.clear()  # Clear previous plots
    self.curr_force_sep_data = np.array(self.all_force_sep_matrices_after_filter[0][0])
    force = self.curr_force_sep_data[:, 1]
    force = force[~np.isnan(force)]
    sep = self.curr_force_sep_data[:, 0]
    sep = sep[~np.isnan(sep)]

    x = sep
    y = force

    self.Force_Sep_Graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3)
    self.Force_Sep_Graph.ax.set_title("Indentation Curve", fontsize=title_size)
    self.Force_Sep_Graph.ax.set_xlabel("separation [m]", fontsize=font_size)
    self.Force_Sep_Graph.ax.set_ylabel("force [N]", fontsize=font_size)
    self.Force_Sep_Graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.Force_Sep_Graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.Force_Sep_Graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.Force_Sep_Graph.ax.tick_params(axis='both', labelsize=font_size)

    self.Force_Sep_Graph.ax.grid(True)
    self.Force_Sep_Graph.fig.tight_layout()
    self.Force_Sep_Graph.draw()


def plot_graph_in_contact_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)

    self.contact_graph.ax.clear()  # Clear previous plots
    self.curr_force_sep_data = np.array(self.all_force_sep_matrices_after_filter_merged[self.curr_force_i])
    d0_values = self.d0
    force_index = self.force_combobox_4.currentIndex()
    ipt_values = self.all_ipt_vectors_merged[force_index]

    i = self.force_curve_list_2.currentRow()
    force = self.curr_force_sep_data[:, i * 2 + 1]
    force = force[~np.isnan(force)]
    force = force[int(ipt_values[i]):-1]
    sep = self.curr_force_sep_data[:, i * 2]
    sep = sep[~np.isnan(sep)]
    sep = sep[int(ipt_values[i]):-1]

    # ipt = int(ipt_values[i])
    # d0 = d0_values[i]
    # length_unit_text = self.R_power_name
    # self.Indentation_depth_label.setText("Indentation Depth = " + str(round(d0*10**(-self.R_power),2))+ " " + length_unit_text)

    # sep2 = [sep[ipt], sep[-1], sep[-1]]
    # force2 = [force[ipt],force[ipt], force[-1]]
    x = sep
    y = force

    self.contact_graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3)
    # self.contact_graph.ax.plot(sep2, force2, marker="o", linestyle="--", linewidth=3)
    self.contact_graph.ax.set_title("Indentation Curve", fontsize=title_size)
    self.contact_graph.ax.set_xlabel("separation [m]", fontsize=font_size)
    self.contact_graph.ax.set_ylabel("force [N]", fontsize=font_size)
    self.contact_graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.contact_graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.contact_graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.contact_graph.ax.tick_params(axis='both', labelsize=font_size)

    self.contact_graph.ax.grid(True)
    self.contact_graph.fig.tight_layout()
    self.contact_graph.draw()


def plot_default_graph_in_contact_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)

    self.contact_graph.ax.clear()  # Clear previous plots
    self.curr_force_sep_data = np.array(self.all_force_sep_matrices_after_filter_merged[0])
    force = self.curr_force_sep_data[:, 1]
    force = force[~np.isnan(force)]
    sep = self.curr_force_sep_data[:, 0]
    sep = sep[~np.isnan(sep)]

    x = sep
    y = force

    self.contact_graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3)
    self.contact_graph.ax.set_title("Indentation Curve", fontsize=title_size)
    self.contact_graph.ax.set_xlabel("separation [m]", fontsize=font_size)
    self.contact_graph.ax.set_ylabel("force [N]", fontsize=font_size)
    self.contact_graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.contact_graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.contact_graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.contact_graph.ax.tick_params(axis='both', labelsize=font_size)

    self.contact_graph.ax.grid(True)
    self.contact_graph.fig.tight_layout()
    self.contact_graph.draw()


def plot_graph_in_analysis_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((24 / 1440) * height)

    check_type = self.check_type_combobox.currentIndex()

    self.Analysis_graph.ax.clear()  # Clear previous plots
    for i in range(self.forcenum):
        d0_value = self.d0_averages_final[i]
        d0_std = self.d0_stds_final[i]
        length_unit_text = self.R_power_name
        labeltext = "$δ_{0}$ = " + str(round(d0_value * 10 ** (-self.R_power), 2)) + " ± " + str(
            round(d0_std * 10 ** (-self.R_power), 2)) + " " + length_unit_text

        if check_type == 0:
            x = self.norm_x_vectors_avg[i]
            y = self.norm_y_vectors_avg[i]
            error = self.norm_y_vectors_std[i]
        elif check_type == 1:
            x = self.visco_x_vectors_avg[i]
            y = self.visco_y_vectors_avg[i]
            error = self.visco_y_vectors_std[i]
        elif check_type == 2:
            x = self.poro_x_vectors_avg[i]
            y = self.poro_y_vectors_avg[i]
            error = self.poro_y_vectors_std[i]
        if check_type == 0 or check_type == 1 or check_type == 2:
            x = np.array(x)
            y = np.array(y)
            error = np.array(error)
            #self.Analysis_graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=2, label=labeltext)
            self.Analysis_graph.ax.plot(x, y, linewidth=2, label=labeltext)
            self.Analysis_graph.ax.fill_between(x, y - error, y + error, alpha=0.2)
    if check_type == 3:
        y1 = self.variance_poro_vector
        x1 = range(len(self.variance_poro_vector))
        y2 = self.variance_visco_vector
        x2 = range(len(self.variance_visco_vector))
        self.Analysis_graph.ax.plot(x1, y1, linewidth=2, label="Poro Test")
        self.Analysis_graph.ax.plot(x2, y2, linewidth=2, label="Visco Test")
    elif check_type == 4:
        y = self.d0_averages_final
        x = self.forcevalues_arr
        y1 = np.array([0.24, 0.34, 0.4, 0.52]) * 10 ** -6
        x1 = np.array([148, 354, 590, 1181]) * 10 ** -9
        error1 = np.array([0.034, 0.032, 0.032, 0.036]) * 10 ** -6
        line1 = self.Analysis_graph.ax.plot(x, y, marker="s", linestyle="--", linewidth=3, label=self.experiment_name,
                                            markersize=15)
        self.Analysis_graph.ax.errorbar(x, y, yerr=self.d0_stds_final, fmt='o', capsize=10, capthick=3, alpha=1.0,
                                        elinewidth=4, ecolor=line1[0].get_color())
        line2 = self.Analysis_graph.ax.plot(x1, y1, marker="s", linestyle="--", linewidth=3,
                                            label="Contact lens,\nR=5.1 μm", markersize=15)
        self.Analysis_graph.ax.errorbar(x1, y1, yerr=error1, fmt='o', capsize=10, capthick=3, alpha=1.0, elinewidth=4,
                                        ecolor=line2[0].get_color())

    if check_type == 0:
        self.Analysis_graph.ax.set_title("Relaxation curves - " + self.experiment_name, fontsize=title_size)
        self.Analysis_graph.ax.set_xlabel("Time [s]", fontsize=font_size)
        self.Analysis_graph.ax.set_ylabel("Force [N]", fontsize=font_size)
        self.Analysis_graph.ax.legend(loc="upper right", fontsize=font_size)
    elif check_type == 1:
        self.Analysis_graph.ax.set_title("Viscoelasticity test - " + self.experiment_name, fontsize=title_size)
        self.Analysis_graph.ax.set_xlabel("Time [s]", fontsize=font_size)
        self.Analysis_graph.ax.set_ylabel("Normalized force", fontsize=font_size)
        self.Analysis_graph.ax.legend(loc="upper right", fontsize=font_size)

    elif check_type == 2:
        self.Analysis_graph.ax.set_title("Poroelasticity test - " + self.experiment_name, fontsize=title_size)
        self.Analysis_graph.ax.set_xlabel("Normalized time [s/m^2]", fontsize=font_size)
        self.Analysis_graph.ax.set_ylabel("Normalized force", fontsize=font_size)
        self.Analysis_graph.ax.legend(loc="upper right", fontsize=font_size)

    elif check_type == 3:
        self.Analysis_graph.ax.set_title("Variance analysis - " + self.experiment_name, fontsize=title_size)
        self.Analysis_graph.ax.set_xlabel("Point #", fontsize=font_size)
        self.Analysis_graph.ax.set_ylabel("Variance", fontsize=font_size)
        self.Analysis_graph.ax.legend(loc="upper right", fontsize=font_size)


    elif check_type == 4:
        self.Analysis_graph.ax.set_title("Stiffness analysis - " + self.experiment_name, fontsize=title_size)
        self.Analysis_graph.ax.set_xlabel("Indentation Force [N]", fontsize=font_size)
        self.Analysis_graph.ax.set_ylabel("Indentation Depth [m]", fontsize=font_size)
        self.Analysis_graph.ax.legend(loc="upper right", fontsize=font_size)

    if (not check_type == 3):
        self.Analysis_graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.Analysis_graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.Analysis_graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.Analysis_graph.ax.tick_params(axis='both', labelsize=font_size)
    self.Analysis_graph.ax.grid(True)
    self.Analysis_graph.fig.tight_layout()
    self.Analysis_graph.draw()


def plot_graph_in_fit_graph(self):
    screen = QApplication.primaryScreen()
    screen_size = screen.size()
    height = screen_size.height()
    font_size = int((18 / 1440) * height)
    title_size = int((32 / 1440) * height)

    i = self.force_fit_combobox.currentIndex()
    if self.fit_type == "visco":
        visco = True
        poro = False
        fit_x_vector = np.array(self.norm_x_vectors_avg[i])
        fit_y_vector = self.visco_y_vectors_avg_fits[i]
        fit_y_vector_UB = self.visco_y_vectors_UB_fits[i]
        fit_y_vector_LB = self.visco_y_vectors_LB_fits[i]
        fit_textbox_text = None
    elif self.fit_type == "poro":
        visco = False
        poro = True
        fit_x_vector = np.array(self.poro_x_vectors_avg[i])
        fit_y_vector = np.array(self.poro_y_vectors_avg_fits[i])
        fit_y_vector_UB = np.array(self.poro_y_vectors_UB_fits[i])
        fit_y_vector_LB = np.array(self.poro_y_vectors_LB_fits[i])
        fit_textbox_text = None

    else:
        return
    self.fit_graph.ax.clear()  # Clear previous plots
    d0_value = self.d0_averages_final[i]
    d0_std = self.d0_stds_final[i]
    length_unit_text = self.R_power_name
    labeltext = "$δ_{0}$ = " + str(round(d0_value * 10 ** (-self.R_power), 2)) + " ± " + str(
        round(d0_std * 10 ** (-self.R_power), 2)) + " " + length_unit_text
    if visco:
        x = self.norm_x_vectors_avg[i]
        y = self.norm_y_vectors_avg[i]
        error = self.norm_y_vectors_std[i]
    elif poro:
        x = self.poro_x_vectors_avg[i]
        y = self.poro_y_vectors_avg[i]
        error = self.poro_y_vectors_std[i]
    else:
        return

    x = np.array(x)
    y = np.array(y)
    error = np.array(error)
    self.fit_graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3, label=labeltext)
    self.fit_graph.ax.fill_between(x, y - error, y + error, alpha=0.2)
    if fit_y_vector is not None:
        if poro:
            if self.conical_indenter:
                fit_text = "Poro function fit, cone"
            else:
                fit_text = "Poro function fit, sphere"
        else:
            fit_text = "Frac viscoelastic function fit"
        self.fit_graph.ax.plot(x, fit_y_vector, marker="None", linestyle="--", linewidth=3, label=fit_text)
        self.fit_graph.ax.plot(x, fit_y_vector_LB, marker="None", linestyle="--", linewidth=3, label=fit_text + " LB")
        self.fit_graph.ax.plot(x, fit_y_vector_UB, marker="None", linestyle="--", linewidth=3, label=fit_text + " UB")
    if visco:
        self.fit_graph.ax.set_title("Viscoelasticity fit - " + self.experiment_name, fontsize=title_size)
        self.fit_graph.ax.set_ylabel("Force [N]", fontsize=font_size)
        self.fit_graph.ax.set_xlabel("Time [s]", fontsize=font_size)
        if fit_textbox_text is not None:
            plt.text(-9, max(y), fit_textbox_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    elif poro:
        self.fit_graph.ax.set_ylabel("Normalized force", fontsize=font_size)
        self.fit_graph.ax.set_title("Poroelasticity test - " + self.experiment_name, fontsize=title_size)
        self.fit_graph.ax.set_xlabel("Normalized time [s/m^2]", fontsize=font_size)
        if fit_textbox_text is not None:
            plt.text(-9, max(y), fit_textbox_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    font_size = int(font_size * (24 / 18))
    self.fit_graph.ax.legend(loc="upper right", fontsize=font_size)
    self.fit_graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
    self.fit_graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
    self.fit_graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
    self.fit_graph.ax.tick_params(axis='both', labelsize=font_size)
    self.fit_graph.ax.grid(True)
    self.fit_graph.fig.tight_layout()
    self.fit_graph.draw()


def cluster_traces(n_clusters: int, force_signal: hs.signals.Signal1D):
    # Clustering
    #force_signal.decomposition(algorithm="SVD") #<-- PCA application
    force_signal.cluster_analysis(cluster_source="signal", n_clusters=n_clusters, preprocessing="norm",
                                  algorithm="kmeans", n_init=8)

    cluster_labels = force_signal.learning_results.cluster_labels[:]
    cluster_labels_list = []
    cluster_labels_list.append(cluster_labels)

    for cluster_id, labels in enumerate(cluster_labels):
        true_indices = [index for index, value in enumerate(labels) if value == True]
        # false_indices = [index for index, value in enumerate(labels) if value == False]
        print(f"In cluster {cluster_id}, graphs {true_indices} belong to the cluster.")
    # save_matrix_as_csv(sep_columns.tolist(), "sep_columns.csv")
    # save_matrix_as_csv(force_columns.tolist(), "force_columns.csv")
    # return sep_columns, force_columns
    return cluster_labels


"""
def cluster_traces_ts(n_clusters: int, force_signal):

    data = force_signal.data

    model = TimeSeriesKMeans(n_clusters=n_clusters,metric="dtw",n_init=2,max_iter=50,random_state=0)

    raw_labels = model.fit_predict(data)

    cluster_labels = np.zeros((n_clusters, len(raw_labels)), dtype=bool)
    for cluster_id in range(n_clusters):
        cluster_labels[cluster_id] = (raw_labels == cluster_id)

    for cluster_id in range(n_clusters):
        true_indices = np.where(cluster_labels[cluster_id] == True)[0].tolist()
        print(f"In cluster {cluster_id}, graphs {true_indices} belong to the cluster.")

    return cluster_labels
"""


def get_true_indices_vector(cluster_labels, cluster_id):
    """
    Returns the indices of elements in the specified cluster that contain `True` values.

    Parameters:
        cluster_labels (list of lists or arrays): A list where each element is a list or array of Boolean values.
        cluster_id (int): The ID of the cluster to process.

    Returns:
        list: Indices of elements in the specified cluster where any value is `True`.
    """
    if cluster_id < 0 or cluster_id >= len(cluster_labels):
        return []
    if cluster_labels.ndim == 2:
        labels = cluster_labels[cluster_id]
    elif cluster_labels.ndim == 1:
        labels = cluster_labels
    else:
        return []

    # Convert to np.ndarray for uniform processing
    labels = np.array(labels, dtype=bool)

    # Collect indices where any value in `value` is True
    true_indices = [index for index, value in enumerate(labels) if value]

    return true_indices


def merge_2d_lists(left_list, right_list):
    left_list = np.array(left_list)
    right_list = np.array(right_list)
    if len(left_list) > 0 and len(right_list) > 0:
        # Merge rows from both lists
        if (left_list.shape[0] > right_list.shape[0]):
            right_list = process_2d_list_specific_length(right_list, left_list.shape[0])
        elif (left_list.shape[0] < right_list.shape[0]):
            left_list = process_2d_list_specific_length(left_list, right_list.shape[0])
        merged_2d_list = np.hstack((left_list, right_list))
    elif len(left_list) > 0:  # Only the left list is non-empty
        merged_2d_list = left_list
    elif len(right_list) > 0:  # Only the right list is non-empty
        merged_2d_list = right_list
    else:  # Both lists are empty
        merged_2d_list = np.array([])
    return merged_2d_list


def make_statistics_vectors(self, input_list):
    matrix = np.array(input_list)
    y_matrix = np.zeros((matrix.shape[0], 0.5 * matrix.shape[1]))
    x_matrix = np.zeros((matrix.shape[0], 0.5 * matrix.shape[1]))

    for i in range(int(0.5 * matrix.shape[0])):
        current_x = matrix[:, 2 * i]
        current_x = current_x[~np.isnan(current_x)]
        current_y = matrix[:, 2 * i + 1]
        current_y = current_y[~np.isnan(current_y)]
        x_matrix[:, i] = current_x
        y_matrix[:, i] = current_y

    y_vector = np.zeros(int((matrix.shape[1])))
    x_vector = np.zeros(int((matrix.shape[1])))

    for i in range(int(y_matrix.shape[0])):
        y_row = y_matrix[i, :]
        y_vector[i] = np.mean(y_row)
        x_row = x_matrix[i, :]
        x_vector[i] = np.mean(x_row)

    return x_vector, y_vector


def Freedman_Diaconis(wave_analyzed):
    """
    Calculates the Freedman-Diaconis bin width for a given dataset.

    Parameters:
    wave_analyzed (numpy.ndarray): The input data array.

    Returns:
    float: The bin width according to the Freedman-Diaconis rule.
    """
    # Compute the interquartile range (IQR)
    IQR = np.percentile(wave_analyzed, 75) - np.percentile(wave_analyzed, 25)

    # Number of data points
    n = len(wave_analyzed)

    # Freedman-Diaconis bin width formula
    bin_width = 2 * IQR * (n ** (-1 / 3))
    if bin_width > 0:  # Ensure bin width is valid
        num_bins = max(1, int((max(wave_analyzed) - min(wave_analyzed)) / bin_width))  # Compute number of bins
    else:
        num_bins = 10  # Default fallback if bin_width is too small

    return num_bins


def save_graph_as_tiff(self, folder_path, filename):
    """Saves the current graph as a TIFF file in the specified folder."""
    if self.FitTab.currentIndex() == 4:
        graph_widget = self.fit_graph
    else:
        graph_widget = self.Analysis_graph

    if folder_path:  # Ensure folder path is valid
        filepath = os.path.join(folder_path, f"{filename}.tiff")
        graph_widget.fig.savefig(filepath, format="tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})

        print(f"Graph saved as: {filepath}")


def interp_after_x_interruption(input_list, x_avg_list):
    matrix = np.array(input_list)
    x_avg_vector = np.array(x_avg_list)

    for i in range(int(0.5 * matrix.shape[1])):
        x_current = matrix[:, 2 * i]
        x_current = x_current[~np.isnan(x_current)]
        force = matrix[:, 2 * i + 1]
        force = force[~np.isnan(force)]
        force_interp = np.interp(x_avg_vector, x_current, force)

        matrix[:, 2 * i] = x_avg_vector
        matrix[:, 2 * i + 1] = force_interp
    return matrix


"""def visco(x, beta, AEC):
    if beta > 0:
        return (AEC / (gamma(1 - beta))) * x ** (-beta)
    else:
        return 0"""
def visco(t, beta, AEC):
    return (AEC / gamma(1 - beta)) * t ** (-beta)

def visco_jac(t, beta, AEC):
    J = np.empty((len(t), 2))
    gamma_val = gamma(1 - beta)
    digamma_val = digamma(1 - beta)

    # ∂/∂beta: product rule with respect to beta
    J[:, 0] = visco(t, beta, AEC) * (
        digamma_val - np.log(t)
    )

    # ∂/∂AEC is straightforward
    J[:, 1] = t ** (-beta) / gamma_val
    return J


def poro_sphere(x, D_p):
    return 0.491 * np.exp(-0.908 * np.sqrt(D_p * x)) + 0.509 * np.exp(-1.679 * D_p * x)


def poro_cone(x, D_p):
    return 0.493 * np.exp(-0.822 * np.sqrt(D_p * x)) + 0.507 * np.exp(-1.348 * D_p * x)

def change_window_size_image_export(self,width, height):
    # Resize to nominal size
    nominal_size = QSize(width, height)
    self.resize(nominal_size)

def restore_window_size(self):
    self.resize(self.original_size)


class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.base_dpi = 100  # A base DPI value that will be adjusted dynamically
        self.fig, self.ax = Figure(dpi=self.base_dpi), None  # Initialize figure with base DPI
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Call the method to adjust the figure size and DPI based on the window size
        self.update_figure_size_and_dpi()
        self.plot_initial_graph()

    def update_figure_size_and_dpi(self):
        """Dynamically update the figure size and adjust DPI based on the window size."""
        available_width = self.size().width()
        available_height = self.size().height()

        dynamic_dpi = max(100, min(available_width // 5, available_height // 5))

        width_in_inches = available_width / dynamic_dpi
        height_in_inches = available_height / dynamic_dpi

        self.fig.set_size_inches(width_in_inches, height_in_inches)
        self.fig.set_dpi(dynamic_dpi)

    def resizeEvent(self, event):
        """Override the resize event to dynamically adjust the figure size and DPI."""
        # self.update_figure_size_and_dpi()
        super().resizeEvent(event)

    def plot_initial_graph(self):
        """Plot a basic initial graph."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(x, y)
        self.ax.set_title("Sample Mathematical Graph", fontsize=18)
        self.ax.set_xlabel("X-axis", fontsize=14)
        self.ax.set_ylabel("Y-axis", fontsize=14)
        self.ax.tick_params(axis='both', which='major', labelsize=18)
        self.draw()

    """def plot_force_curve(self, sep, force, sep2, force2, title):
        self.ax.clear()
        self.ax.plot(sep, force, 'r', label='Force')
        self.ax.plot(sep2, force2, 'b--o', label='Processed')
        self.ax.set_title(title, fontsize=18)
        self.ax.set_xlabel('Separation (um)', fontsize=14)
        self.ax.set_ylabel('Force (nN)', fontsize=14)
        self.ax.legend()
        self.ax.tick_params(axis='both', which='major', labelsize=14)
        self.draw()"""




class FindD0App(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        #uic.loadUi('form2.ui', self)
        self.setupUi(self)
        self.setStyleSheet("""
            QWidget {
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
        self.base_width = 500
        self.base_font_pt = 14

        self.setWindowIcon(QIcon("Logo.ico"))  # Replace "icon.ico" with your icon file path
        self.setWindowTitle("DigiGel Analysis")
        screen = QDesktopWidget().screenGeometry()
        screen_x = ((screen.width() - self.width()) // 2) - (screen.width() // 13)  # Center horizontally
        screen_y = 0 * (screen.height() - self.height()) // 4  # Place higher (1/4th from top)

        self.move(screen_x, screen_y)

        self.curr_force_sep_data = None
        self.d0 = None
        self.all_d0_vectors = []
        self.all_a_vectors = []
        self.d0_default = None
        self.all_ipts = []
        #print("Assigned self.all_ipts at 1736:", self.all_ipts, type(self.all_ipts))
        self.def_all_ipts = []
        self.all_ipt_vectors = []
        self.all_d0_vectors_merged = []
        self.all_a_vectors_merged = []
        self.all_ipt_vectors_merged = []
        self.all_E_star_vectors_merged = []
        self.forcenum = 0
        self.curr_force_i = 0
        self.curr_sample_i = 0
        self.duplicatenum = 0
        self.R_micron = 0
        self.n_clusters = 0
        self.final_trace_number = 0
        self.current_model_index = 0
        self.current_model = ""
        self.traceNames = []
        self.graphtitle_arr = []
        self.forcevalues_arr = []
        self.all_force_sep_AI_indices = []
        self.all_force_time_AI_indices = []
        self.all_force_sep_matrices_before_filter = []
        self.all_force_time_matrices_before_filter = []
        self.all_index_vectors_after_filtering = []
        self.all_force_sep_matrices_after_filter = []
        self.all_force_time_matrices_after_filter = []
        self.all_index_vectors_after_filtering_merged = []
        self.all_force_sep_matrices_after_filter_merged = []
        self.all_force_time_matrices_after_filter_merged = []
        self.current_force_sep_matrix = []
        self.current_force_time_matrix = []
        self.norm_x_vectors_avg = []
        self.visco_x_vectors_avg = []
        self.poro_x_vectors_avg = []
        self.norm_y_vectors_avg = []
        self.visco_y_vectors_avg = []
        self.poro_y_vectors_avg = []
        self.norm_x_vectors_std = []
        self.visco_x_vectors_std = []
        self.poro_x_vectors_std = []
        self.norm_y_vectors_std = []
        self.visco_y_vectors_std = []
        self.poro_y_vectors_std = []
        self.norm_x_vectors_UB = []
        self.visco_x_vectors_UB = []
        self.poro_x_vectors_UB = []
        self.norm_y_vectors_UB = []
        self.visco_y_vectors_UB = []
        self.poro_y_vectors_UB = []
        self.norm_x_vectors_LB = []
        self.visco_x_vectors_LB = []
        self.poro_x_vectors_LB = []
        self.norm_y_vectors_LB = []
        self.visco_y_vectors_LB = []
        self.poro_y_vectors_LB = []
        self.norm_all_matrices = []
        self.visco_all_matrices = []
        self.poro_all_matrices = []
        self.force_sep_uploaded_flag = False
        self.force_time_uploaded_flag = False
        self.current_force_signal = None
        self.current_sep_columns = None
        self.current_force_columns = None
        self.debug = False
        self.current_list_of_list_of_clusters = []
        self.current_index_vector = []
        self.F_power = 1
        self.R_power = 1
        self.F_power_name = ""
        self.R_power_name = ""
        self.d0_averages_divided = []
        self.d0_averages_final = []
        self.d0_stds_final = []
        self.current_force_sep_CSV_length = 0
        self.current_force_time_CSV_length = 0
        self.debug_upload_csv_i = 0
        self.variance_visco_vector = []
        self.variance_poro_vector = []
        self.variance_poro_vectors = []
        self.experiment_name = ""
        self.conical_indenter = False
        self.spherical_indenter = False
        self.half_angle_indenter_rad = 0
        self.tan_alpha_indenter = 0
        self.current_tab = 0
        self.trace_names_after_filtering_merged_sep = []
        self.trace_names_after_filtering_merged_time = []
        self.D_p = []
        self.G = []
        self.nu = []
        self.beta = []
        self.C_beta = []
        self.D_p_std = []
        self.G_std = []
        self.nu_std = []
        self.beta_std = []
        self.C_beta_std = []
        self.D_p_tot = 0
        self.G_tot = 0
        self.nu_tot = 0
        self.beta_tot = 0
        self.C_beta_tot = 0
        self.D_p_std_tot = 0
        self.G_std_tot = 0
        self.nu_std_tot = 0
        self.beta_std_tot = 0
        self.C_beta_std_tot = 0
        self.r_squared_visco = 0
        self.r_squared_poro = 0
        self.fit_type = ""
        self.visco_y_vectors_avg_fits = []
        self.poro_y_vectors_avg_fits = []
        self.visco_y_vectors_UB_fits = []
        self.poro_y_vectors_UB_fits = []
        self.visco_y_vectors_LB_fits = []
        self.poro_y_vectors_LB_fits = []
        self.all_G_values = []
        self.all_nu_values = []
        self.poro_fits_performed_already = False
        self.visco_fits_performed_already = False
        self.r_squared_visco_values = []
        self.r_squared_poro_values = []
        self.restart = False
        self.saving_version = 1
        self.manual_filtering_flag = False
        self.all_headers_vectors_before_filter = []
        self.all_headers_vectors_taken_out = []
        self.filtered_out_indices = []
        self.filtered_out_indices_reason = []
        self.dark_mode_enabled = False
        self.count_to_upload = 0
        self.max_sep_val = 0
        self.min_sep_val = 0
        self.settle_time_sec = 0
        self.original_size = self.size()
        self.camefromtab2 = False

        self.force_curve_list = self.findChild(QListWidget, 'force_curve_list')
        self.force_curve_list_2 = self.findChild(QListWidget, 'force_curve_list_2')
        self.load_ForceSep_csv_button = self.findChild(QPushButton, 'load_ForceSep_csv_button')
        self.load_ForceTime_csv_button = self.findChild(QPushButton, 'load_ForceTime_csv_button')
        self.calc_d0_button = self.findChild(QPushButton, 'calc_d0_button')
        self.d0_slider = self.findChild(QSlider, 'd0_slider')
        self.contact_slider = self.findChild(QSlider, 'contact_slider')
        self.enter_data_btn = self.findChild(QPushButton, 'enter_data_btn')
        self.force_num_combobox = self.findChild(QComboBox, 'force_num_combobox')
        self.duplicate_num_combobox = self.findChild(QComboBox, 'duplicate_num_combobox')
        self.force_values_input = self.findChild(QDoubleSpinBox, 'force_values_input')
        self.enter_force_nN_btn = self.findChild(QPushButton, 'enter_force_nN_btn')
        self.enter_force_label = self.findChild(QLabel, 'enter_force_label')
        self.load_ForceSep_csv_button = self.findChild(QPushButton, 'load_ForceSep_csv_button')
        self.load_ForceTime_csv_button = self.findChild(QPushButton, 'load_ForceTime_csv_button')
        self.next_tab_0_button = self.findChild(QPushButton, "next_tab_0_button")
        self.force_combobox_1 = self.findChild(QComboBox, 'force_combobox_1')
        self.sample_combobox_1 = self.findChild(QComboBox, 'sample_combobox_1')
        self.force_combobox_2 = self.findChild(QComboBox, 'force_combobox_2')
        self.sample_combobox_2 = self.findChild(QComboBox, 'sample_combobox_2')
        self.gridLayout = self.findChild(QGridLayout, 'gridLayout')
        self.gridLayout_2 = self.findChild(QGridLayout, 'gridLayout_2')
        self.horizontalLayout_5 = self.findChild(QHBoxLayout, 'horizontalLayout_5')
        self.R_micron_spinbox = self.findChild(QDoubleSpinBox, 'R_micron_spinbox')
        self.Header_label = self.findChild(QLabel, 'Header_label')
        self.DigiGel = self.findChild(QLabel, 'DigiGel')
        self.copyright_label = self.findChild(QLabel, 'copyright_label')
        # self.update_instance_button = self.findChild(QPushButton, 'update_instance_button')
        self.elbow_check = self.findChild(QPushButton, 'elbow_check')
        self.force_sep_UL_radio_button = self.findChild(QRadioButton, 'force_sep_UL_radio_button')
        self.force_time_UL_radio_button = self.findChild(QRadioButton, 'force_time_UL_radio_button')
        self.cluster_num_label = self.findChild(QLabel, 'cluster_num_label')
        self.apply_clustreing_button = self.findChild(QPushButton, 'apply_clustreing_button')
        self.Cluster_Number_Combobox = self.findChild(QComboBox, 'Cluster_Number_Combobox')
        self.clusterComboBox = self.findChild(QComboBox, 'clusterComboBox')
        self.graphTraceAIComboBox = self.findChild(QComboBox, 'graphTraceAIComboBox')
        self.erase_trace_button = self.findChild(QPushButton, 'erase_trace_button')
        self.erase_cluster_button = self.findChild(QPushButton, 'erase_cluster_button')
        self.start_over_AI_button = self.findChild(QPushButton, 'start_over_AI_button')
        self.filtering_accepted_button = self.findChild(QPushButton, 'filtering_accepted_button')
        self.Indentation_depth_label = self.findChild(QLabel, 'Indentation_depth_label')
        self.default_d0_button = self.findChild(QPushButton, 'default_d0_button')
        self.Save_d0_button = self.findChild(QPushButton, 'Save_d0_button')
        self.model_find_d0_combobox = self.findChild(QComboBox, 'model_find_d0_combobox')
        self.Change_model_label = self.findChild(QLabel, 'Change_model_label')
        self.meter_scale_combobox = self.findChild(QComboBox, 'meter_scale_combobox')
        self.newton_scale_combobox = self.findChild(QComboBox, 'newton_scale_combobox')
        self.check_type_combobox = self.findChild(QComboBox, 'check_type_combobox')
        self.show_histogram_button = self.findChild(QPushButton, 'show_histogram_button')
        self.Lowest_button = self.findChild(QPushButton, 'Lowest_button')
        self.Highest_button = self.findChild(QPushButton, 'Highest_button')
        self.skip_d0_button = self.findChild(QPushButton, 'skip_d0_button')
        self.debug_button = self.findChild(QPushButton, 'debug_button')
        self.experiment_name_lineedit = self.findChild(QLineEdit, 'experiment_name_lineedit')
        self.indenter_type_combobox = self.findChild(QComboBox, 'indenter_type_combobox')
        self.Indenter_Label = self.findChild(QLabel, 'Indenter_Label')
        self.save_graphs_button = self.findChild(QPushButton, 'save_graphs_button')
        self.save_experiment_button = self.findChild(QPushButton, 'save_experiment_button')
        self.load_experiment_button = self.findChild(QPushButton, 'load_experiment_button')
        self.save_filtered_data_button = self.findChild(QPushButton, 'save_filtered_data_button')
        self.verticalLayout_9 = self.findChild(QVBoxLayout, 'verticalLayout_9')
        self.verticalLayout_10 = self.findChild(QVBoxLayout, 'verticalLayout_10')
        self.parameter_label_1 = self.findChild(QLabel, 'parameter_label_1')
        self.parameter_label_2 = self.findChild(QLabel, 'parameter_label_2')
        self.parameter_label_3 = self.findChild(QLabel, 'parameter_label_3')
        self.parameter_label_4 = self.findChild(QLabel, 'parameter_label_4')
        self.current_parameter_label_1 = self.findChild(QLabel, 'current_parameter_label_1')
        self.current_parameter_label_2 = self.findChild(QLabel, 'current_parameter_label_2')
        self.current_parameter_label_3 = self.findChild(QLabel, 'current_parameter_label_3')
        self.current_parameter_label_4 = self.findChild(QLabel, 'current_parameter_label_4')
        self.force_fit_combobox = self.findChild(QComboBox, 'force_fit_combobox')
        self.make_poro_fits_button = self.findChild(QPushButton, 'make_poro_fits_button')
        self.make_visco_fits_button = self.findChild(QPushButton, 'make_visco_fits_button')
        self.save_analyzed_data_button = self.findChild(QPushButton, 'save_analyzed_data_button')
        self.save_fit_graph_button = self.findChild(QPushButton, 'save_fit_graph_button')
        self.save_fit_values_button = self.findChild(QPushButton, 'save_fit_values_button')
        self.start_over_button = self.findChild(QPushButton, 'start_over_button')
        self.manual_filtering_button = self.findChild(QPushButton, 'manual_filtering_button')
        self.remove_current_trace_d0_button = self.findChild(QPushButton, 'remove_current_trace_d0_button')
        self.force_combobox_4 = self.findChild(QComboBox, 'force_combobox_4')
        self.label_20 = self.findChild(QLabel, 'label_20')
        # self.calc_contact_button = self.findChild(QPushButton, 'calc_contact_button')
        self.contact_model_combobox = self.findChild(QComboBox, 'contact_model_combobox')
        self.adhesion_force_label = self.findChild(QLabel, 'adhesion_force_label')
        self.adhesion_force_combobox = self.findChild(QComboBox, 'adhesion_force_combobox')
        self.adhesion_force_spinbox = self.findChild(QSpinBox, 'adhesion_force_spinbox')
        self.light_dark_mode_button = self.findChild(QPushButton, 'light_dark_mode_button')
        self.loading_label = self.findChild(QLabel, 'loading_label')
        self.fit_for_current_force_label = self.findChild(QLabel, 'fit_for_current_force_label')
        self.fit_for_all_force_label = self.findChild(QLabel, 'fit_for_all_force_label')
        self.settle_time_spinbox = self.findChild(QDoubleSpinBox, 'settle_time_spinbox')
        self.save_clusters_csv_button = self.findChild(QPushButton, 'save_clusters_csv_button')

        self.load_ForceSep_csv_button.clicked.connect(self.load_new_forcesep_csv)
        self.load_ForceTime_csv_button.clicked.connect(self.load_new_forcetime_csv)
        self.calc_d0_button.clicked.connect(self.calc_d0_button_clicked)
        self.force_curve_list.itemSelectionChanged.connect(self.force_curve_list_updated)
        #        self.force_curve_list_2.itemSelectionChanged.connect(self.force_curve_list_2_updated)
        self.d0_slider.valueChanged.connect(self.update_d0_slider)
        #        self.contact_slider.valueChanged.connect(self.update_contact_slider)
        self.enter_data_btn.clicked.connect(self.enter_data_btn_pressed)
        self.enter_force_nN_btn.clicked.connect(self.enter_force_nN_btn_pressed)
        self.next_tab_0_button.clicked.connect(self.next_tab_0_button_pressed)
        #        self.update_instance_button.clicked.connect(self.update_instance)
        self.elbow_check.clicked.connect(self.elbow_check_clicked)
        self.apply_clustreing_button.clicked.connect(self.apply_clustreing_button_clicked)
        self.clusterComboBox.currentIndexChanged.connect(self.clusterComboBox_changed)
        self.graphTraceAIComboBox.currentIndexChanged.connect(self.graphTraceAIComboBox_changed)
        self.erase_trace_button.clicked.connect(self.erase_trace_button_clicked)
        self.erase_cluster_button.clicked.connect(self.erase_cluster_button_clicked)
        self.start_over_AI_button.clicked.connect(self.start_over_AI_button_clicked)
        self.filtering_accepted_button.clicked.connect(self.filtering_accepted_button_clicked)
        self.force_combobox_1.currentIndexChanged.connect(self.force_combobox_1_changed)
        self.sample_combobox_1.currentIndexChanged.connect(self.sample_combobox_1_changed)
        self.force_combobox_2.currentIndexChanged.connect(self.force_combobox_2_changed)
        #        self.force_combobox_4.currentIndexChanged.connect(self.force_combobox_4_changed)
        self.sample_combobox_2.currentIndexChanged.connect(self.sample_combobox_2_changed)
        self.default_d0_button.clicked.connect(self.default_d0_button_clicked)
        self.Save_d0_button.clicked.connect(self.Save_d0_button_clicked)
        self.model_find_d0_combobox.currentIndexChanged.connect(self.model_find_d0_combobox_changed)
        self.check_type_combobox.currentIndexChanged.connect(self.check_type_combobox_changed)
        self.show_histogram_button.clicked.connect(self.show_histogram_button_clicked)
        self.Lowest_button.clicked.connect(self.Lowest_button_clicked)
        self.Highest_button.clicked.connect(self.Highest_button_clicked)
        # self.skip_d0_button.clicked.connect(self.skip_d0_button_clicked)
        # self.debug_button.clicked.connect(self.debug_button_clicked)
        self.indenter_type_combobox.currentIndexChanged.connect(self.indenter_type_combobox_changed)
        self.save_graphs_button.clicked.connect(self.save_graphs_button_clicked)
        self.save_experiment_button.clicked.connect(self.save_experiment_button_clicked)
        self.load_experiment_button.clicked.connect(self.load_experiment_button_clicked)
        self.save_filtered_data_button.clicked.connect(self.save_filtered_data_button_clicked)
        self.make_poro_fits_button.clicked.connect(self.make_poro_fits_button_clicked)
        self.make_visco_fits_button.clicked.connect(self.make_visco_fits_button_clicked)
        self.force_fit_combobox.currentIndexChanged.connect(self.force_fit_combobox_changed)
        self.save_analyzed_data_button.clicked.connect(self.save_analyzed_data_button_clicked)
        self.force_sep_UL_radio_button.clicked.connect(self.force_sep_UL_radio_button_clicked)
        self.force_time_UL_radio_button.clicked.connect(self.force_time_UL_radio_button_clicked)
        self.save_fit_graph_button.clicked.connect(self.save_fit_graph_button_clicked)
        self.save_fit_values_button.clicked.connect(self.save_fit_values_button_clicked)
        self.start_over_button.clicked.connect(self.start_over_button_clicked)
        self.manual_filtering_button.clicked.connect(self.manual_filtering_button_clicked)
        self.remove_current_trace_d0_button.clicked.connect(self.remove_current_trace_d0_button_clicked)
        #        self.calc_contact_button.clicked.connect(self.calc_contact_button_clicked)
        #        self.contact_model_combobox.currentIndexChanged.connect(self.contact_model_combobox_changed)
        self.light_dark_mode_button.clicked.connect(self.light_dark_mode_button_clicked)
        self.save_clusters_csv_button.clicked.connect(self.save_clusters_csv_button_clicked)
        self.fit_for_current_force_label.setVisible(False)
        self.fit_for_all_force_label.setVisible(False)
        self.current_parameter_label_1.setVisible(False)
        self.current_parameter_label_2.setVisible(False)
        self.current_parameter_label_3.setVisible(False)
        self.current_parameter_label_4.setVisible(False)
        self.parameter_label_1.setVisible(False)
        self.parameter_label_2.setVisible(False)
        self.parameter_label_3.setVisible(False)
        self.parameter_label_4.setVisible(False)
        """font = QFont()
        font.setPointSize(13)  # adjust as needed
        self.parameter_label_1.setFont(font)
        self.parameter_label_2.setFont(font)
        self.parameter_label_3.setFont(font)
        self.parameter_label_4.setFont(font)
        font.setBold(True)  # adjust as needed
        self.current_parameter_label_1.setFont(font)
        self.current_parameter_label_2.setFont(font)
        self.current_parameter_label_3.setFont(font)
        self.current_parameter_label_4.setFont(font)"""

        """screen = QApplication.primaryScreen()
        screen_size = screen.size()
        window_size_ratio = 0.6
        self.resize(int(screen_size.width() * window_size_ratio), int(screen_size.height() * window_size_ratio))"""
        screen = QApplication.primaryScreen()
        logical_geom = screen.availableGeometry()
        width = int(logical_geom.width() * 0.6)
        height = int(logical_geom.height() * 0.95)
        self.resize(width, height)
        self.setMinimumSize(200, 200)  # or an appropriate lower bound

        # Replacing the placeholder widget with MatplotlibCanvas
        self.Force_Sep_Graph = MatplotlibCanvas(self)
        self.horizontalLayout_5.replaceWidget(self.findChild(QtWidgets.QWidget, 'Force_Sep_Graph'),
                                              self.Force_Sep_Graph)
        self.AI_Viewing_Graph = MatplotlibCanvas(self)
        self.gridLayout_2.replaceWidget(self.findChild(QtWidgets.QWidget, 'AI_Viewing_Graph'), self.AI_Viewing_Graph)
        self.Analysis_graph = MatplotlibCanvas(self)
        self.verticalLayout_6.replaceWidget(self.findChild(QtWidgets.QWidget, 'Analysis_graph'), self.Analysis_graph)
        self.fit_graph = MatplotlibCanvas(self)
        self.verticalLayout_9.replaceWidget(self.findChild(QtWidgets.QWidget, 'fit_graph'), self.fit_graph)
        #        self.contact_graph = MatplotlibCanvas(self)
        #        self.verticalLayout_10.replaceWidget(self.findChild(QtWidgets.QWidget, 'contact_graph'), self.contact_graph)
        # self.force_sep_graph.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # self.AI_Viewing_Graph.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.FitTab.setCurrentIndex(0)  # Assuming the tab index is 2 for the "Find_D0_Tab"
        self.enter_data_btn.setDefault(True)

        self.FitTab.setTabEnabled(1, False)  # Assuming the tab index is 2 for the "Find_D0_Tab"
        self.FitTab.setTabEnabled(2, False)  # Assuming the tab index is 2 for the "Find_D0_Tab"
        self.FitTab.setTabEnabled(3, False)  # Assuming the tab index is 2 for the "Find_D0_Tab"
        self.FitTab.setTabEnabled(4, False)  # Assuming the tab index is 2 for the "Find_D0_Tab"
        self.FitTab.setTabEnabled(5, False)  # Assuming the tab index is 2 for the "Find_D0_Tab"

        QTimer.singleShot(0, self.connect_screen_changed_signal)
        #self.debug_button_clicked()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        print(f"{self.width()}, {self.height()}")
        # Calculate how much bigger or smaller the window is compared to the base width
        scale_factor = self.width() / self.base_width

        # Compute a new font size
        new_font_pt = int(self.base_font_pt * scale_factor)

        # Prevent the font from getting too tiny or too huge
        new_font_pt = max(4, min(new_font_pt, 24))

        # Apply to the entire window
        font = self.font()
        font.setPointSize(new_font_pt)
        self.setFont(font)

    def closeEvent(self, event):
        """Prompt the user with an 'Are you sure?' message before closing."""
        if self.restart:
            event.accept()
        else:
            reply = QMessageBox.question(self, "Exit Confirmation",
                                         "Save experiment?",
                                         QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)

            if reply == QMessageBox.No:
                event.accept()  # Close the application
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                is_saved = self.save_experiment_button_clicked()
                if not is_saved:
                    event.ignore()

    def keyPressEvent(self, event: QKeyEvent):
        """Detects key presses and executes actions."""
        if event.key() == Qt.Key_Q:
            if self.current_tab == 1:
                self.erase_trace_button_clicked()
        elif event.key() == Qt.Key_C:
            if self.current_tab == 1:
                self.erase_cluster_button_clicked()
        elif event.key() == Qt.Key_L:
            if self.current_tab == 0:
                self.load_experiment_button_clicked()
            elif self.current_tab == 1:
                self.filtering_accepted_button_clicked()
        elif event.key() == Qt.Key_H:
            if self.current_tab == 2:
                self.show_histogram_button_clicked()
        elif event.key() == Qt.Key_Down:
            if self.current_tab == 1:
                i = self.graphTraceAIComboBox.currentIndex()
                if not i == self.graphTraceAIComboBox.count() - 1:
                    self.graphTraceAIComboBox.setCurrentIndex(i + 1)
        elif event.key() == Qt.Key_Up:
            if self.current_tab == 1:
                i = self.graphTraceAIComboBox.currentIndex()
                if not i == 0:
                    self.graphTraceAIComboBox.setCurrentIndex(i - 1)
        elif event.key() == Qt.Key_Right:
            if self.current_tab == 2:
                if self.d0_slider.isEnabled():
                    self.d0_slider.setValue(self.d0_slider.value() + self.d0_slider.singleStep())
        elif event.key() == Qt.Key_Left:
            if self.current_tab == 2:
                if self.d0_slider.isEnabled():
                    self.d0_slider.setValue(self.d0_slider.value() - self.d0_slider.singleStep())
        elif event.key() == Qt.Key_Enter or event.key() == 16777220:  # enter key
            if self.current_tab == 0:
                if self.enter_data_btn.isEnabled():
                    self.enter_data_btn_pressed()
                elif self.enter_force_nN_btn.isEnabled():
                    self.enter_force_nN_btn_pressed()
                elif self.next_tab_0_button.isEnabled():
                    self.next_tab_0_button_pressed()
            elif self.current_tab == 2:
                if self.calc_d0_button.isEnabled():
                    self.calc_d0_button_clicked()
                elif self.Save_d0_button.isEnabled():
                    self.calc_d0_button_clicked()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self.save_experiment_button_clicked()
        elif event.key() == Qt.Key_9 and event.modifiers() & Qt.ControlModifier:
            if self.current_tab==0:
                self.debug_button_clicked()
            elif self.current_tab==2:
                self.skip_d0_button_clicked()

            """elif self.current_tab == 1:
                if self.elbow_check.isEnabled():
                    self.elbow_check_clicked()
                elif self.apply_clustreing_button.isEnabled():
                    self.apply_clustreing_button_clicked()
                elif self.filtering_accepted_button.isEnabled():
                    self.filtering_accepted_button_clicked()"""


        elif event.key() == Qt.Key_Escape:
            self.close()

        """elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.label.setText("Enter Pressed!")
        else:
            self.label.setText(f"Key Pressed: {event.text()}")  # Show pressed key"""

    def connect_screen_changed_signal(self):
        """Connect screenChanged signal after the window has been fully initialized."""
        if self.windowHandle():
            self.windowHandle().screenChanged.connect(self.handle_screen_change)

    def handle_screen_change(self):
        screen = QApplication.screenAt(self.pos())
        if screen:
            screen_size = screen.size()
            window_size_ratio = 0.6
            self.resize(int(screen_size.width() * window_size_ratio), int(screen_size.height() * window_size_ratio))

    def load_new_forcesep_csv(self):
        """Load CSV data and handle NaN values."""
        file_dialog = QFileDialog(self)
        if self.debug:
            if self.debug_upload_csv_i == 0:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_25nN_sep.csv'
                file_path = 'E1_25nN_sep.csv'
            elif self.debug_upload_csv_i == 1:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_25nN_sep.csv'
                file_path = 'E1_25nN_sep.csv'
            elif self.debug_upload_csv_i == 2:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_90nN_sep.csv'
                file_path = 'E1_90nN_sep.csv'
            elif self.debug_upload_csv_i == 3:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E2_90nN_sep.csv'
                file_path = 'E2_90nN_sep.csv'
            elif self.debug_upload_csv_i == 4:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_400nN_sep.csv'
                file_path = 'E1_400nN_sep.csv'
            elif self.debug_upload_csv_i == 5:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E2_400nN_sep.csv'
                file_path = 'E2_400nN_sep.csv'
        else:
            file_path, _ = file_dialog.getOpenFileName(self, "Load Indentation CSV", "", "CSV files (*.csv)")

        if file_path:
            currentlist = np.genfromtxt(file_path, delimiter=',', skip_header=1);
            current_length = currentlist.shape[1]
            if not self.current_force_time_CSV_length == 0 and not self.current_force_time_CSV_length == current_length:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Indentation and Relaxation trace number does not match')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return

            if (currentlist.shape[0] < 20 or currentlist.shape[1] < 20):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Too Little Data')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return
            self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i]  = process_2d_list(currentlist)
            # self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i] = fill_empty_space_2D_list(self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i], find_maximum_length_column(self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i]))
            self.force_sep_uploaded_flag = True
            self.current_force_sep_CSV_length = current_length
            self.load_ForceSep_csv_button.setStyleSheet(
                """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")

            if (self.force_time_uploaded_flag * self.force_sep_uploaded_flag):
                self.next_tab_0_button.setEnabled(True)

    def next_tab_0_button_pressed(self):
        self.load_ForceSep_csv_button.setStyleSheet(
            """background-color: #FDFDFD; color: black; font-size: 14pt;""")
        self.load_ForceTime_csv_button.setStyleSheet(
            """background-color: #FDFDFD; color: black; font-size: 14pt;""")
        self.count_to_upload = self.count_to_upload - 1
        if not self.count_to_upload == 0:
            self.next_tab_0_button.setText("Next (" + str(self.count_to_upload) + " to go)")
        else:
            self.next_tab_0_button.setText("Go to next tab")


        self.load_ForceSep_csv_button.setStyleSheet("")
        self.load_ForceTime_csv_button.setStyleSheet("")
        self.current_force_sep_CSV_length = 0
        self.current_force_time_CSV_length = 0
        if self.curr_sample_i < self.duplicatenum - 1:
            self.curr_sample_i = self.curr_sample_i + 1;
            self.update_csv_buttons_text()
            self.force_time_uploaded_flag = False
            self.force_sep_uploaded_flag = False
        else:
            if self.curr_force_i < self.forcenum - 1:
                self.curr_force_i = self.curr_force_i + 1
                self.curr_sample_i = 0
                if (self.force_time_uploaded_flag * self.force_sep_uploaded_flag):
                    self.update_csv_buttons_text()
                    self.force_time_uploaded_flag = False
                    self.force_sep_uploaded_flag = False
            else:
                if (self.curr_force_i == self.forcenum - 1 and self.curr_sample_i == self.duplicatenum - 1):
                    self.curr_force_i = 0
                    self.curr_sample_i = 0
                    self.load_ForceSep_csv_button.setEnabled(False)
                    self.load_ForceTime_csv_button.setEnabled(False)
                    self.FitTab.setCurrentIndex(1)
                    self.FitTab.setTabEnabled(1, True)
                    self.FitTab.setTabEnabled(0, False)
                    current_data_matrix = self.all_force_time_matrices_before_filter[self.curr_force_i][
                        self.curr_sample_i]
                    current_y_vector = [row[1] for row in current_data_matrix]
                    current_x_vector = [row[0] for row in current_data_matrix]
                    plot_graph_in_ai_viewing_graph(self, current_x_vector, current_y_vector, True)

                    self.save_experiment_button.setEnabled(True)
                    self.load_experiment_button.setEnabled(False)
                    self.initialize_tab_1()
                    self.current_tab = 1
                    self.max_sep_val, self.min_sep_val = get_even_column_extrema(self.all_force_sep_matrices_before_filter)



                else:
                    return
        self.next_tab_0_button.setEnabled(False)

    def update_csv_buttons_text(self):
        self.load_ForceSep_csv_button.setEnabled(True)
        self.load_ForceSep_csv_button.setText(
            "Load force vs. separation\nindentation phase:\nsample #" + str(self.curr_sample_i + 1) + ", " + str(
                round(self.forcevalues_arr[self.curr_force_i] * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)
        self.load_ForceTime_csv_button.setEnabled(True)
        self.load_ForceTime_csv_button.setText(
            "Load force vs. time\nrelaxation phase:\nsample #" + str(self.curr_sample_i + 1) + ", " + str(
                round(self.forcevalues_arr[self.curr_force_i] * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)

    def load_new_forcetime_csv(self):
        """Load CSV data and handle NaN values."""
        file_dialog = QFileDialog(self)
        if self.debug:
            if self.debug_upload_csv_i == 0:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_25nN_time.csv'
                file_path = 'E1_25nN_time.csv'
            elif self.debug_upload_csv_i == 1:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_25nN_time.csv'
                file_path = 'E1_25nN_time.csv'
            elif self.debug_upload_csv_i == 2:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_90nN_time.csv'
                file_path = 'E1_90nN_time.csv'
            elif self.debug_upload_csv_i == 3:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E2_90nN_time.csv'
                file_path = 'E2_90nN_time.csv'
            elif self.debug_upload_csv_i == 4:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E1_400nN_time.csv'
                file_path = 'E1_400nN_time.csv'
            elif self.debug_upload_csv_i == 5:
                # file_path = 'C:/Users/beckerm/OneDrive - post.bgu.ac.il/PhD/Research/Protocols/DigiGel Installation/DigiGel_Software/E2_400nN_time.csv'
                file_path = 'E2_400nN_time.csv'
        else:
            file_path, _ = file_dialog.getOpenFileName(self, "Load Relaxation CSV", "", "CSV files (*.csv)")

        if file_path:
            if not self.current_force_time_CSV_length == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Indentation and Relaxation trace number does not match')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return

            currentlist = np.genfromtxt(file_path, delimiter=',', skip_header=1);
            current_length = currentlist.shape[1]
            df = pd.read_csv(file_path)
            self.all_headers_vectors_before_filter[self.curr_force_i][self.curr_sample_i] = list(df.columns)[::2]
            if not self.current_force_sep_CSV_length == 0 and not self.current_force_sep_CSV_length == current_length:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Indentation and Relaxation trace number does not match')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return

            if (currentlist.shape[0] < 20 or currentlist.shape[1] < 20):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Too Little Data')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return
            currentlist_to_normalize_x_vals = process_2d_list(currentlist)
            self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i] = Normalize_x_val_2d_forcetime_list(currentlist_to_normalize_x_vals)

            matrix = adapt_to_controller_settle_time(self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i], self.settle_time_sec)
            self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i] = matrix
            """self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i] = fill_empty_space_2D_list(
                self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i],
                find_maximum_length_column(
                    self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i]))"""
            # self.traceNames = [f"Trace {i // 2 + 1}" for i in range(0, len(self.curr_force_sep_data[0]), 2)]
            # self.force_curve_list.clear()
            # self.force_curve_list.addItems(self.traceNames)
            self.force_time_uploaded_flag = True
            self.current_force_time_CSV_length = current_length
            self.load_ForceTime_csv_button.setStyleSheet(
                """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")

        if (self.force_time_uploaded_flag * self.force_sep_uploaded_flag):
            self.next_tab_0_button.setEnabled(True)

    """def calculate_d0(self):
        start = time.perf_counter()
        data = np.array(self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i])
        traces = data.shape[1] // 2
        model = self.model_find_d0_combobox.currentText()

        args = [(i, data, model) for i in range(traces)]

        with ThreadPoolExecutor() as executor:
            results = executor.map(calculate_d0_for_trace, args)

        self.d0 = np.zeros(traces)
        self.all_ipts = np.zeros(traces)
        self.def_all_ipts = np.zeros(traces)

        for i, ipt, def_ipt, d0_val in results:
            self.all_ipts[i] = ipt
            self.def_all_ipts[i] = def_ipt
            self.d0[i] = d0_val

        self.d0_default = self.d0
        # (Update scale units here if needed...)
        print(f"Time taken: {time.perf_counter() - start:.2f} seconds")"""

    def calculate_d0(self):
        start = time.perf_counter()
        self.curr_force_sep_data = np.array(
            self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i])
        traces = self.curr_force_sep_data.shape[1] // 2
        self.d0 = np.zeros(traces)
        self.all_ipts = np.zeros(traces)
        #print("Assigned self.all_ipts at 2442:", self.all_ipts, type(self.all_ipts))
        self.def_all_ipts = np.zeros(traces)

        for i in range(traces):
            force = self.curr_force_sep_data[:, i * 2 + 1]
            force = force[~np.isnan(force)]
            sep = self.curr_force_sep_data[:, i * 2]
            sep = sep[~np.isnan(sep)]
            ipt = find_ipt(self, force)


            d0 = (sep[ipt] - sep[-1])

            if d0 <= 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Calculation Error")
                msg.setText("A trace yielded d0<=0, so the software cannot continue. Check your data.")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return False

            self.all_ipts[i] = ipt
            self.def_all_ipts[i] = ipt
            self.d0[i] = d0
        all_ipts_def = self.all_ipts
        self.d0_default = self.d0
        if self.conical_indenter:
            log_average_d0 = np.log10(np.mean(np.array(self.d0)))

            if log_average_d0 < 0 and log_average_d0 > -3:
                self.R_power = -3
                self.R_power_name = "mm"
            elif log_average_d0 < -3 and log_average_d0 > -6:
                self.R_power = -6
                self.R_power_name = "μm"
            elif log_average_d0 < -6 and log_average_d0 > -9:
                self.R_power = -9
                self.R_power_name = "nm"
            else:
                self.R_power = 0
                self.R_power_name = "m"
        print(f"Time taken: {time.perf_counter() - start:.2f} seconds")
        return True

    def initiate_slider(self):
        self.curr_force_sep_data = np.array(
            self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i])
        i = self.force_curve_list.currentRow()
        sep = self.curr_force_sep_data[:, i * 2]
        sep = sep[~np.isnan(sep)]
        self.d0_slider.setMinimum(1)  # Set the minimum value of the slider
        self.d0_slider.setMaximum(len(sep))  # Set the maximum value of the slider
        if len(self.all_ipts)==0:
            self.all_ipts = [0]
        if not isinstance(self.all_ipts, int):
            ipt = int(self.all_ipts[i])
            self.d0_slider.setValue(len(sep) - ipt)  # Set the value of the slider
        # self.update_d0_slider(len(sep)-ipt)

    def enter_data_btn_pressed(self):
        """The Enter button is pressed."""
        self.settle_time_sec = self.settle_time_spinbox.value()
        self.forcenum = self.force_num_combobox.currentIndex() + 2
        self.duplicatenum = self.duplicate_num_combobox.currentIndex() + 1
        self.count_to_upload = self.forcenum*self.duplicatenum-1
        self.all_force_sep_matrices_before_filter = [[0 for i in range(self.duplicatenum)] for j in
                                                     range(self.forcenum)]
        self.all_force_time_matrices_before_filter = [[0 for i in range(self.duplicatenum)] for j in
                                                      range(self.forcenum)]
        self.all_force_sep_matrices_after_filter = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_force_time_matrices_after_filter = [[0 for i in range(self.duplicatenum)] for j in
                                                     range(self.forcenum)]
        self.all_force_sep_AI_indices = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_force_time_AI_indices = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_index_vectors_after_filtering = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_d0_vectors = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_a_vectors = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_ipt_vectors = [[[0] for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.d0_averages_divided = [[0 for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_headers_vectors_before_filter = [[[] for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.all_headers_vectors_taken_out = [[[] for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.filtered_out_indices = [[[] for i in range(self.duplicatenum)] for j in range(self.forcenum)]
        self.filtered_out_indices_reason = [[[] for i in range(self.duplicatenum)] for j in range(self.forcenum)]

        self.all_force_sep_matrices_after_filter_merged = [0 for i in range(self.forcenum)]
        self.all_force_time_matrices_after_filter_merged = [0 for i in range(self.forcenum)]
        self.d0_averages_final = [0 for i in range(self.forcenum)]
        self.d0_stds_final = [0 for i in range(self.forcenum)]
        self.norm_all_matrices = [0 for i in range(self.forcenum)]
        self.visco_all_matrices = [0 for i in range(self.forcenum)]
        self.poro_all_matrices = [0 for i in range(self.forcenum)]
        self.norm_x_vectors_avg = [0 for i in range(self.forcenum)]
        self.visco_x_vectors_avg = [0 for i in range(self.forcenum)]
        self.poro_x_vectors_avg = [0 for i in range(self.forcenum)]
        self.norm_y_vectors_avg = [0 for i in range(self.forcenum)]
        self.visco_y_vectors_avg = [0 for i in range(self.forcenum)]
        self.poro_y_vectors_avg = [0 for i in range(self.forcenum)]
        self.norm_x_vectors_std = [0 for i in range(self.forcenum)]
        self.visco_x_vectors_std = [0 for i in range(self.forcenum)]
        self.poro_x_vectors_std = [0 for i in range(self.forcenum)]
        self.norm_y_vectors_std = [0 for i in range(self.forcenum)]
        self.visco_y_vectors_std = [0 for i in range(self.forcenum)]
        self.poro_y_vectors_std = [0 for i in range(self.forcenum)]
        self.norm_x_vectors_UB = [0 for i in range(self.forcenum)]
        self.visco_x_vectors_UB = [0 for i in range(self.forcenum)]
        self.poro_x_vectors_UB = [0 for i in range(self.forcenum)]
        self.norm_y_vectors_UB = [0 for i in range(self.forcenum)]
        self.visco_y_vectors_UB = [0 for i in range(self.forcenum)]
        self.poro_y_vectors_UB = [0 for i in range(self.forcenum)]
        self.norm_x_vectors_LB = [0 for i in range(self.forcenum)]
        self.visco_x_vectors_LB = [0 for i in range(self.forcenum)]
        self.poro_x_vectors_LB = [0 for i in range(self.forcenum)]
        self.norm_y_vectors_LB = [0 for i in range(self.forcenum)]
        self.visco_y_vectors_LB = [0 for i in range(self.forcenum)]
        self.poro_y_vectors_LB = [0 for i in range(self.forcenum)]
        self.all_d0_vectors_merged = [[] for i in range(self.forcenum)]
        self.all_a_vectors_merged = [[] for i in range(self.forcenum)]
        self.all_ipt_vectors_merged = [[] for i in range(self.forcenum)]

        self.all_E_star_vectors_merged = [[] for i in range(self.forcenum)]
        #        self.variance_visco_vector = [[] for i in range(self.forcenum)]
        #       self.variance_poro_vector = [[] for i in range(self.forcenum)]
        self.all_index_vectors_after_filtering_merged = [[] for i in range(self.forcenum)]
        self.trace_names_after_filtering_merged_sep = [[] for i in range(self.forcenum)]
        self.trace_names_after_filtering_merged_time = [[] for i in range(self.forcenum)]
        self.variance_poro_vectors = [[] for i in range(self.forcenum)]
        self.experiment_name = self.experiment_name_lineedit.text()
        self.D_p = [0 for i in range(self.forcenum)]
        self.G = [0 for i in range(self.forcenum)]
        self.nu = [0 for i in range(self.forcenum)]
        self.beta = [0 for i in range(self.forcenum)]
        self.C_beta = [0 for i in range(self.forcenum)]
        self.D_p_std = [0 for i in range(self.forcenum)]
        self.G_std = [0 for i in range(self.forcenum)]
        self.nu_std = [0 for i in range(self.forcenum)]
        self.beta_std = [0 for i in range(self.forcenum)]
        self.C_beta_std = [0 for i in range(self.forcenum)]
        self.visco_y_vectors_avg_fits = [[] for i in range(self.forcenum)]
        self.poro_y_vectors_avg_fits = [[] for i in range(self.forcenum)]
        self.visco_y_vectors_UB_fits = [[] for i in range(self.forcenum)]
        self.poro_y_vectors_UB_fits = [[] for i in range(self.forcenum)]
        self.visco_y_vectors_LB_fits = [[] for i in range(self.forcenum)]
        self.poro_y_vectors_LB_fits = [[] for i in range(self.forcenum)]
        self.all_G_values = [[] for i in range(self.forcenum)]
        self.all_nu_values = [[] for i in range(self.forcenum)]
        self.r_squared_visco_values = [0 for i in range(self.forcenum)]
        self.r_squared_poro_values = [0 for i in range(self.forcenum)]

        self.duplicate_num_combobox.setEnabled(False)
        self.force_num_combobox.setEnabled(False)
        self.enter_data_btn.setEnabled(False)
        self.settle_time_spinbox.setEnabled(False)
        self.enter_force_nN_btn.setEnabled(True)
        self.force_values_input.setEnabled(True)
        self.meter_scale_combobox.setEnabled(False)
        self.newton_scale_combobox.setEnabled(True)
        indenter_type_i = self.indenter_type_combobox.currentIndex()
        self.experiment_name_lineedit.setEnabled(False)
        self.indenter_type_combobox.setEnabled(False)

        if indenter_type_i == 0:
            self.conical_indenter = False
            self.spherical_indenter = True
            self.R_micron = self.R_micron_spinbox.value()
            self.R_power = (-3) * self.meter_scale_combobox.currentIndex()
            self.R_power_name = self.meter_scale_combobox.currentText()
            self.R_micron = self.R_micron * 10 ** self.R_power

        elif indenter_type_i == 1:
            self.conical_indenter = True
            self.spherical_indenter = False
            self.half_angle_indenter_rad = self.R_micron_spinbox.value()
            if self.meter_scale_combobox.currentIndex() == 0:
                self.half_angle_indenter_rad = np.radians(self.half_angle_indenter_rad)
            self.tan_alpha_indenter = np.tan(self.half_angle_indenter_rad)
        else:
            raise ValueError("Incorrect indenter type")
        self.R_micron_spinbox.setEnabled(False)

    def enter_force_nN_btn_pressed(self):
        """The Enter force button is pressed."""
        if (self.curr_force_i < self.forcenum - 1):
            self.F_power = (-3) * self.newton_scale_combobox.currentIndex()
            self.F_power_name = self.newton_scale_combobox.currentText()
            force_value_N = self.force_values_input.value() * 10 ** self.F_power
            if force_value_N == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Force cannot be zero')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return
            self.forcevalues_arr.append(double(force_value_N))
            self.force_values_input.setValue(0)
            self.curr_force_i = self.curr_force_i + 1
            self.enter_force_label.setText("Enter Force #" + str(self.curr_force_i + 1) + ":")
        else:
            force_value_N = self.force_values_input.value() * 10 ** self.F_power
            if force_value_N == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('Force cannot be zero')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return
            self.forcevalues_arr.append(double(self.force_values_input.value() * 10 ** self.F_power))
            self.newton_scale_combobox.setEnabled(False)
            self.enter_force_nN_btn.setEnabled(False)
            self.force_values_input.setEnabled(False)
            self.curr_force_i = 0
            self.load_ForceSep_csv_button.setEnabled(True)
            self.load_ForceSep_csv_button.setText("Load force vs. separation\nindentation phase:\nsample #1, " + str(
                round(self.forcevalues_arr[0] * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)
            self.load_ForceTime_csv_button.setEnabled(True)
            self.load_ForceTime_csv_button.setText("Load force vs. time\nrelaxation phase:\nSample #1, " + str(
                round(self.forcevalues_arr[0] * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)
            self.next_tab_0_button.setText("Next (" + str(self.count_to_upload) + " to go)")

            # self.next_tab_0_button.setEnabled(True)
        self.newton_scale_combobox.setEnabled(False)

    def update_d0_slider(self, value):
        self.curr_force_sep_data = np.array(
            self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i])
        i = self.force_curve_list.currentRow()
        sep = self.curr_force_sep_data[:, i * 2]
        sep = sep[~np.isnan(sep)]
        if not (value <= 1 or value >= len(sep)):
            ipt = len(sep) - value
            print("all_ipts:", self.all_ipts, type(self.all_ipts))
            self.all_ipts[i] = ipt
        else:
            if self.all_ipts[i]==0:
                self.all_ipts[i]=1
            ipt = int(self.all_ipts[i])
        d0 = (sep[ipt] - sep[-1])
        self.d0[i] = d0
        self.Indentation_depth_label.setText("Indentation Depth = " + str(round(d0, 2)) + " μm")
        plot_graph_in_find_d0_graph(self)

    def update_contact_slider(self, value):
        self.curr_force_sep_data = np.array(self.all_force_sep_matrices_after_filter_merged[self.curr_force_i])
        i = self.force_curve_list.currentRow()
        force = self.curr_force_sep_data[:, i * 2 + 1]
        force = force[~np.isnan(force)]
        sep = self.curr_force_sep_data[:, i * 2]
        sep = sep[~np.isnan(sep)]
        ipt = self.all_ipt_vectors_merged[self.curr_force_i][i]
        # number_of_points_in_indentation_region = force.length[0]-ipt
        self.label_20.setText(str(value) + "%")
        plot_graph_in_contact_graph(self)

    def initialize_tab_1(self):
        for label in self.forcevalues_arr:
            self.force_combobox_1.addItem(str(round(label * 10 ** (-self.F_power), 1)) + self.F_power_name)
        for label in range(1, self.duplicatenum + 1):
            self.sample_combobox_1.addItem(str(label))
        self.update_current_matrices()

    def initialize_tab_2(self):
        self.FitTab.setTabEnabled(1, False)
        self.FitTab.setTabEnabled(2, True)
        self.force_combobox_2.clear()
        self.sample_combobox_2.clear()
        for label in self.forcevalues_arr:
            self.force_combobox_2.addItem(str(round(label * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)
        for label in range(1, self.duplicatenum + 1):
            self.sample_combobox_2.addItem(str(label))
        self.update_current_matrices_filtered()
        self.curr_force_i = 0
        self.curr_sample_i = 0
        current_indices = self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i]
        self.traceNames = [f"Trace {i + 1}" for i in current_indices]
        self.force_curve_list.clear()
        self.force_curve_list.addItems(self.traceNames)
        plot_default_graph_in_find_d0_graph(self)

    def initialize_tab_5(self):
        self.curr_force_i = 0
        self.force_combobox_4.clear()
        for label in self.forcevalues_arr:
            self.force_combobox_4.addItem(str(round(label * 10 ** (-self.F_power), 1)) + " " + self.F_power_name)
        current_indices = self.all_index_vectors_after_filtering_merged[self.curr_force_i]
        self.traceNames = [f"Trace {i}" for i in current_indices]
        self.force_curve_list_2.clear()
        self.force_curve_list_2.addItems(self.traceNames)
        plot_default_graph_in_contact_graph(self)

    def update_current_matrices(self):
        self.current_force_sep_matrix = self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i]
        self.current_force_time_matrix = self.all_force_time_matrices_before_filter[self.curr_force_i][
            self.curr_sample_i]

    def update_current_matrices_filtered(self):
        self.current_force_sep_matrix = self.all_force_sep_matrices_after_filter[self.curr_force_i][self.curr_sample_i]
        self.current_force_time_matrix = self.all_force_time_matrices_after_filter[self.curr_force_i][
            self.curr_sample_i]

    def force_sep_UL_radio_button_clicked(self):
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):
            self.filtering_accepted_button.setEnabled(True)

    def force_time_UL_radio_button_clicked(self):
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):
            self.filtering_accepted_button.setEnabled(True)

    def sample_combobox_1_changed(self):
        #        self.update_instance_button.setStyleSheet(
        #            """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):
            self.filtering_accepted_button.setEnabled(True)
        self.curr_force_i = self.force_combobox_1.currentIndex()
        self.curr_sample_i = self.sample_combobox_1.currentIndex()
        self.force_combobox_1.setCurrentIndex(0)
        self.update_instance()

    def force_combobox_1_changed(self):
        #        self.update_instance_button.setStyleSheet(
        #            """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        # if (not isinstance(current_indices, int)):
        #    self.filtering_accepted_button.setEnabled(True)

        self.curr_force_i = self.force_combobox_1.currentIndex()
        self.curr_sample_i = self.sample_combobox_1.currentIndex()
        self.force_sep_UL_radio_button.setChecked(True)
        self.update_instance()

    def sample_combobox_2_changed(self):
        # self.calc_d0_button.setStyleSheet(
        #    """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")
        self.curr_force_i = self.force_combobox_2.currentIndex()
        self.curr_sample_i = self.sample_combobox_2.currentIndex()
        self.all_ipts = self.all_ipt_vectors[self.curr_force_i][self.curr_sample_i]
        if isinstance(self.all_ipts, int):
            self.all_ipts = [0]
        #print("Assigned self.all_ipts at 2808:", self.all_ipts, type(self.all_ipts))
        current_indices = self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i]
        self.traceNames = [f"Trace {i+1}" for i in current_indices]
        self.force_curve_list.clear()
        self.force_curve_list.addItems(self.traceNames)
        self.force_combobox_2.setCurrentIndex(0)

    def force_combobox_2_changed(self):
        # self.calc_d0_button.setStyleSheet(
        #    """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")
        self.curr_force_i = self.force_combobox_2.currentIndex()
        self.curr_sample_i = self.sample_combobox_2.currentIndex()
        self.all_ipts = self.all_ipt_vectors[self.curr_force_i][self.curr_sample_i]
        if isinstance(self.all_ipts, int):
            self.all_ipts = [0]
        #print("Assigned self.all_ipts at 2821:", self.all_ipts, type(self.all_ipts))
        current_indices = self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i]
        if not isinstance(current_indices, int):
            self.traceNames = [f"Trace {i+1}" for i in current_indices]
        self.force_curve_list.clear()
        self.force_curve_list.addItems(self.traceNames)

    def force_combobox_4_changed(self):
        # self.calc_d0_button.setStyleSheet(
        #    """background-color: #90EE90; color: black; font-size: 14pt;font-weight: bold;""")
        self.curr_force_i = self.force_combobox_4.currentIndex()
        current_indices = self.all_index_vectors_after_filtering_merged[self.curr_force_i]
        if not isinstance(current_indices, int):
            self.traceNames = [f"Trace {i}" for i in current_indices]
        self.force_curve_list_2.clear()
        self.force_curve_list_2.addItems(self.traceNames)

    def update_instance(self):
        self.curr_force_i = self.force_combobox_1.currentIndex()
        self.curr_sample_i = self.sample_combobox_1.currentIndex()
        self.update_current_matrices()
        self.force_sep_UL_radio_button.setEnabled(True)
        self.force_time_UL_radio_button.setEnabled(True)
        self.elbow_check.setEnabled(True)
        self.manual_filtering_button.setEnabled(True)
        # self.update_instance_button.setEnabled(False)
        # self.update_instance_button.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5() + """
        #    QWidget {
        #        font-family: 'Arial', sans-serif;
        #        font-size: 22px;
        #    }
        #    QPushButton {
        #        font-family: 'Arial';
        #        font-size: 14pt;
        #    }
        # """)

    def calc_d0_button_clicked(self):
        current_a_vector = self.all_a_vectors[self.curr_force_i][self.curr_sample_i]
        if (not isinstance(current_a_vector, int)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("This instance is already calculated, do you want to override it?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()

            if response == QMessageBox.No:
                return

        is_Okay = self.calculate_d0()
        if is_Okay:
            self.force_curve_list.setEnabled(True)
            self.force_curve_list.setCurrentRow(0)
            self.calc_d0_button.setEnabled(False)
            self.curr_force_i = self.force_combobox_2.currentIndex()
            self.curr_sample_i = self.sample_combobox_2.currentIndex()
            self.sample_combobox_2.setEnabled(False)
            self.force_combobox_2.setEnabled(False)
            self.update_current_matrices_filtered()
            self.elbow_check.setEnabled(True)
            self.manual_filtering_button.setEnabled(True)
            self.calc_d0_button.setEnabled(False)
            self.d0_slider.setEnabled(True)
            self.default_d0_button.setEnabled(True)
            self.Save_d0_button.setEnabled(True)
            # self.model_find_d0_combobox.setEnabled(True)
            self.Indentation_depth_label.setEnabled(True)
            self.Change_model_label.setEnabled(True)
            self.show_histogram_button.setEnabled(True)
            self.Lowest_button.setEnabled(True)
            self.Highest_button.setEnabled(True)
            self.remove_current_trace_d0_button.setEnabled(True)
            self.force_curve_list.setCurrentRow(-1)
            self.show_histogram_button_clicked()

    def manual_filtering_button_clicked(self):
        self.start_over_AI_button.setText("Restart filtering")
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("This instance is already filtered, do you want to override it?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
            if response == QMessageBox.No:
                self.filtering_accepted_button.setEnabled(True)
                self.sample_combobox_1.setEnabled(True)
                self.force_combobox_1.setEnabled(True)
                return
        if (self.force_sep_UL_radio_button.isChecked()):
            a = 3
        elif (self.force_time_UL_radio_button.isChecked()):
            a = 3
        else:
            return
        self.manual_filtering_flag = True
        self.apply_clustreing_button_clicked()

    def elbow_check_clicked(self):
        self.start_over_AI_button.setText("Restart filtering")
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("This instance is already filtered, do you want to override it?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
            if response == QMessageBox.No:
                self.filtering_accepted_button.setEnabled(True)
                self.sample_combobox_1.setEnabled(True)
                self.force_combobox_1.setEnabled(True)
                return
        if (self.force_sep_UL_radio_button.isChecked()):
            current_np_matrix = self.current_force_sep_matrix
            self.n_clusters, self.current_sep_columns, self.current_force_columns = calculate_cluster_number(
                np.array(current_np_matrix), self.AI_Viewing_Graph, False)
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_np_matrix = self.current_force_time_matrix
            self.n_clusters, self.current_sep_columns, self.current_force_columns = calculate_cluster_number(
                np.array(current_np_matrix), self.AI_Viewing_Graph, True)
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        self.manual_filtering_flag = False
        plot_elbow_in_ai_viewing_graph(self, self.current_force_columns)
        self.Cluster_Number_Combobox.setEnabled(True)
        self.Cluster_Number_Combobox.setCurrentIndex(self.n_clusters - 1)
        # self.cluster_num_label.setText("Number of clusters: " + str(self.n_clusters))
        self.apply_clustreing_button.setEnabled(True)
        #######Try to plot the elbow plot in the GUI itself...?
        # self.current_force_signal.plot_cluster_metric()

        self.sample_combobox_1.setEnabled(False)
        self.force_combobox_1.setEnabled(False)
        # self.update_instance_button.setEnabled(False)
        self.elbow_check.setEnabled(False)
        self.manual_filtering_button.setEnabled(False)

    def apply_clustreing_button_clicked(self):
        self.current_force_signal = hs.signals.Signal1D(self.current_force_columns)
        self.all_d0_vectors[self.curr_force_i][self.curr_sample_i] = 0
        if self.manual_filtering_flag:
            self.n_clusters = 1

        else:
            self.n_clusters = self.Cluster_Number_Combobox.currentIndex() + 1;

        self.Cluster_Number_Combobox.setEnabled(False)
        self.apply_clustreing_button.setEnabled(False)
        self.clusterComboBox.setEnabled(True)
        self.graphTraceAIComboBox.setEnabled(True)
        self.elbow_check.setEnabled(False)
        self.manual_filtering_button.setEnabled(False)
        self.force_sep_UL_radio_button.setEnabled(False)
        self.force_time_UL_radio_button.setEnabled(False)
        self.save_clusters_csv_button.setEnabled(True)

        if (self.force_sep_UL_radio_button.isChecked()):
            if self.manual_filtering_flag:
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = np.array(
                    [True for i in range(int(len(
                        self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i][1]) / 2))])
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = \
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i].reshape(1, -1)
            else:
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = cluster_traces(self.n_clusters,
                                                                                                      self.current_force_signal)
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            if self.manual_filtering_flag:
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = np.array(
                    [True for i in range(int(len(
                        self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i][1]) / 2))])
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = \
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i].reshape(1, -1)
            else:
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = cluster_traces(self.n_clusters,
                                                                                                       self.current_force_signal)
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        self.current_index_vector = get_true_indices_vector(current_indices, 0)
        for label in range(0, self.n_clusters):
            self.clusterComboBox.addItem(
                str(label + 1) + " (n=" + str(len(get_true_indices_vector(current_indices, label))) + ")")

        # for label in self.current_index_vector:
        # self.graphTraceAIComboBox.addItem(str(label+1))
        self.erase_trace_button.setEnabled(True)
        if not self.manual_filtering_flag:
            self.erase_cluster_button.setEnabled(True)
        self.start_over_AI_button.setEnabled(True)
        self.filtering_accepted_button.setEnabled(True)

    def clusterComboBox_changed(self):
        current_cluster_index = self.clusterComboBox.currentIndex()
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        if (not isinstance(current_indices, int)):
            self.current_index_vector = get_true_indices_vector(current_indices, current_cluster_index)
            self.graphTraceAIComboBox.clear()
            for label in self.current_index_vector:
                self.graphTraceAIComboBox.addItem(str(label + 1))

    def graphTraceAIComboBox_changed(self):
        current_cluster_index = self.clusterComboBox.currentIndex()
        current_text = str(self.graphTraceAIComboBox.currentText().strip())
        if current_text.strip():  # Check if the text is not empty
            current_trace_index = int(current_text.strip()) - 1
        else:
            return
        if (self.force_sep_UL_radio_button.isChecked()):
            isrelax = False
            current_data_matrix = self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            isrelax = True
            current_data_matrix = self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i]
        else:
            return
        # current_data_matrix_T = [list(row) for row in zip(*current_data_matrix)]

        current_y_vector = [row[2 * current_trace_index + 1] for row in current_data_matrix]
        current_x_vector = [row[2 * current_trace_index] for row in current_data_matrix]
        plot_graph_in_ai_viewing_graph(self, current_x_vector, current_y_vector, isrelax)

    def erase_trace_button_clicked(self):
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        current_cluster_index = self.clusterComboBox.currentIndex()
        current_text = str(self.graphTraceAIComboBox.currentText().strip())
        if current_text.strip():  # Check if the text is not empty
            current_trace_index = int(current_text.strip()) - 1
        else:
            return
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirmation")
        msg.setText("Delete single trace: Are you sure?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # Show the message box and get the response
        font = QFont()
        font.setPointSize(14)  # adjust as needed
        msg.setFont(font)
        response = msg.exec_()

        if response == QMessageBox.Yes:
            self.all_headers_vectors_taken_out[self.curr_force_i][self.curr_sample_i].append(
                self.all_headers_vectors_before_filter[self.curr_force_i][self.curr_sample_i][current_trace_index])
            if self.force_time_UL_radio_button.isChecked():
                self.filtered_out_indices_reason[self.curr_force_i][self.curr_sample_i].append(
                    "Faulty relaxation measurement - manually removed")
            else:
                self.filtered_out_indices_reason[self.curr_force_i][self.curr_sample_i].append(
                    "Faulty indentation measurement - manually removed")
            for row in current_indices:
                row[current_trace_index] = False
            if (self.force_sep_UL_radio_button.isChecked()):
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = current_indices
            elif (self.force_time_UL_radio_button.isChecked()):
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = current_indices
            n_c = self.clusterComboBox.currentIndex()
            n_t = self.graphTraceAIComboBox.currentIndex()
            self.clusterComboBox.clear()
            for label in range(0, self.n_clusters):
                self.clusterComboBox.addItem(
                    str(label + 1) + " (n=" + str(len(get_true_indices_vector(current_indices, label))) + ")")
            self.clusterComboBox_changed()
            self.clusterComboBox.setCurrentIndex(n_c)
            self.graphTraceAIComboBox.setCurrentIndex(n_t)
        else:
            return

    def remove_current_trace_d0_button_clicked(self):
        current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        current_item = self.force_curve_list.currentItem()
        if current_item is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('No Trace Selected')
            msg.setWindowTitle("Error")
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            msg.exec_()
            return

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirmation")
        msg.setText("Delete single trace: Are you sure?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # Show the message box and get the response
        font = QFont()
        font.setPointSize(14)  # adjust as needed
        msg.setFont(font)
        response = msg.exec_()
        if response == QMessageBox.Yes:
            current_text = str(self.force_curve_list.currentItem().text())
            if current_text.strip():  # Check if the text is not empty
                current_trace_index = int(''.join(filter(str.isdigit, current_text))) - 1
            self.all_headers_vectors_taken_out[self.curr_force_i][self.curr_sample_i].append(
                self.all_headers_vectors_before_filter[self.curr_force_i][self.curr_sample_i][current_trace_index])
            self.filtered_out_indices_reason[self.curr_force_i][self.curr_sample_i].append("Faulty indentation measurement - manually removed")
            self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i].remove(current_trace_index)
            self.handle_data_after_single_remove_from_d0_panel()
            current_indices = self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i]
            current_list_index = self.force_curve_list.currentRow()
            self.all_ipts = np.delete(self.all_ipts, current_list_index)
            self.d0 = np.delete(self.d0, current_list_index)
            self.def_all_ipts = np.delete(self.def_all_ipts, current_list_index)
            self.d0_default = np.delete(self.d0_default, current_list_index)

            self.traceNames = [f"Trace {i + 1}" for i in current_indices]
            self.force_curve_list.clear()
            self.force_curve_list.addItems(self.traceNames)
            plot_default_graph_in_find_d0_graph(self)

    def erase_cluster_button_clicked(self):
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        current_cluster_index = self.clusterComboBox.currentIndex()
        current_text = str(self.graphTraceAIComboBox.currentText().strip())
        if current_text.strip():  # Check if the text is not empty
            current_trace_index = int(current_text.strip()) - 1
        else:
            return
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirmation")
        msg.setText("Delete group: Are you sure?")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        # Show the message box and get the response
        font = QFont()
        font.setPointSize(14)  # adjust as needed
        msg.setFont(font)
        response = msg.exec_()

        if response == QMessageBox.Yes:
            true_indices = get_true_indices_vector(current_indices, self.clusterComboBox.currentIndex())
            for k in range(len(true_indices)):
                self.all_headers_vectors_taken_out[self.curr_force_i][self.curr_sample_i].append(
                    self.all_headers_vectors_before_filter[self.curr_force_i][self.curr_sample_i][true_indices[k]])
                if self.force_time_UL_radio_button.isChecked():
                    self.filtered_out_indices_reason[self.curr_force_i][self.curr_sample_i].append(
                        "Faulty relaxation measurement - UL group removed")
                else:
                    self.filtered_out_indices_reason[self.curr_force_i][self.curr_sample_i].append(
                        "Faulty indentation measurement - UL group removed")
            current_indices[current_cluster_index, :] = False
            if (self.force_sep_UL_radio_button.isChecked()):
                self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = current_indices
            elif (self.force_time_UL_radio_button.isChecked()):
                self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = current_indices
            self.clusterComboBox.clear()
            for label in range(0, self.n_clusters):
                self.clusterComboBox.addItem(
                    str(label + 1) + " (n=" + str(len(get_true_indices_vector(current_indices, label))) + ")")
            self.clusterComboBox_changed()
        else:
            return

    def start_over_AI_button_clicked(self):
        if self.start_over_AI_button.text() == "Skip filter":
            for i in range(self.forcenum):
                for j in range(self.duplicatenum):
                    self.all_force_sep_AI_indices[i][j] = np.array(
                        [True for i in range(int(len(self.all_force_sep_matrices_before_filter[i][j][1]) / 2))])
                    self.all_force_time_AI_indices[i][j] = np.array(
                        [True for i in range(int(len(self.all_force_time_matrices_before_filter[i][j][1]) / 2))])
                    self.all_force_sep_AI_indices[i][j] = self.all_force_sep_AI_indices[i][j].reshape(1, -1)
                    self.all_force_time_AI_indices[i][j] = self.all_force_time_AI_indices[i][j].reshape(1, -1)
            length = 0
            for i in range(self.forcenum):
                for j in range(self.duplicatenum):
                    self.all_index_vectors_after_filtering[i][j] = range(
                        int(len(self.all_force_sep_matrices_before_filter[i][j][1]) / 2))
                    length += len(self.all_index_vectors_after_filtering[i][j])

            self.handle_data_after_AI_filter()
            self.FitTab.setCurrentIndex(2)
            self.current_tab = 2
            self.initialize_tab_2()

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("Restart filtering: Are you sure?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()

            if response == QMessageBox.Yes:
                if (self.force_sep_UL_radio_button.isChecked()):
                    self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i] = 0
                elif (self.force_time_UL_radio_button.isChecked()):
                    self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i] = 0
                else:
                    return
                self.sample_combobox_1.setEnabled(True)
                self.force_combobox_1.setEnabled(True)
                # self.update_instance_button.setEnabled(True)
                self.force_sep_UL_radio_button.setEnabled(True)
                self.force_time_UL_radio_button.setEnabled(True)
                self.elbow_check.setEnabled(True)
                self.manual_filtering_button.setEnabled(True)
                self.clusterComboBox.setEnabled(False)
                self.graphTraceAIComboBox.setEnabled(False)
                self.erase_trace_button.setEnabled(False)
                self.erase_cluster_button.setEnabled(False)
                self.start_over_AI_button.setEnabled(False)
                self.filtering_accepted_button.setEnabled(False)
                self.clusterComboBox.clear()
                self.graphTraceAIComboBox.clear()


            else:
                return

    def filtering_accepted_button_clicked(self):
        if not self.debug:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("Save filtering: Are you sure?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
        else:
            response = QMessageBox.Yes

        if response == QMessageBox.Yes:
            self.sample_combobox_1.setEnabled(True)
            self.force_combobox_1.setEnabled(True)
            # self.update_instance_button.setEnabled(True)
            self.elbow_check.setEnabled(True)
            self.manual_filtering_button.setEnabled(True)
            self.force_sep_UL_radio_button.setEnabled(True)
            self.force_time_UL_radio_button.setEnabled(True)
            self.clusterComboBox.setEnabled(False)
            self.graphTraceAIComboBox.setEnabled(False)
            self.erase_trace_button.setEnabled(False)
            self.erase_cluster_button.setEnabled(False)
            self.start_over_AI_button.setEnabled(False)
            self.filtering_accepted_button.setEnabled(False)
            self.save_clusters_csv_button.setEnabled(False)
            self.clusterComboBox.clear()
            self.graphTraceAIComboBox.clear()
            for i in range(self.forcenum):
                for j in range(self.duplicatenum):
                    transient_indices_force_sep = self.all_force_sep_AI_indices[i][j]
                    transient_indices_force_time = self.all_force_time_AI_indices[i][j]
                    if (isinstance(transient_indices_force_sep, int) or isinstance(transient_indices_force_time, int)):
                        return
            if not self.debug:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Confirmation")
                msg.setText("All instances filtered! Do you want to move on to the next step?")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

                # Show the message box and get the response
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)
                response = msg.exec_()
            else:
                response = QMessageBox.Yes

            if response == QMessageBox.Yes:
                self.handle_data_after_AI_filter()
                self.filtering_accepted_button.setEnabled(False)
                self.FitTab.setCurrentIndex(2)
                self.FitTab.setTabEnabled(1, False)
                self.current_tab = 2
                self.initialize_tab_2()
            else:
                self.filtering_accepted_button.setEnabled(True)
                return




        else:
            return

    def handle_data_after_AI_filter(self):
        maximum_trace_number = 0
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                transient_indices_force_sep = self.all_force_sep_AI_indices[i][j]
                transient_indices_force_time = self.all_force_time_AI_indices[i][j]
                if (isinstance(transient_indices_force_sep, int) or isinstance(transient_indices_force_time, int)):
                    return
                transient_indices_force_sep_vector = np.logical_or.reduce(self.all_force_sep_AI_indices[i][j], axis=0)
                transient_indices_force_time_vector = np.logical_or.reduce(self.all_force_time_AI_indices[i][j], axis=0)
                current_indices = transient_indices_force_sep_vector & transient_indices_force_time_vector
                if current_indices.size > maximum_trace_number:
                    maximum_trace_number = current_indices.size
                self.all_index_vectors_after_filtering[i][j] = get_true_indices_vector(current_indices, 0)

        minimum_trace_number = maximum_trace_number
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                current_indices = self.all_index_vectors_after_filtering[i][j]
                if minimum_trace_number > len(current_indices):
                    minimum_trace_number = len(current_indices)
        self.final_trace_number = minimum_trace_number

        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                current_indices = self.all_index_vectors_after_filtering[i][j]
                while len(current_indices) > minimum_trace_number:
                    integer_to_delete = random.randint(0, (len(current_indices) - 1))
                    del current_indices[integer_to_delete]
                    self.all_headers_vectors_taken_out[i][j].append(
                        self.all_headers_vectors_before_filter[i][j][integer_to_delete])
                    self.filtered_out_indices_reason[i][j].append("Randomly removed")
                self.all_index_vectors_after_filtering[i][j] = current_indices
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                current_indices = np.array(self.all_index_vectors_after_filtering[i][j])
                self.all_force_sep_matrices_after_filter[i][j] = select_double_columns(
                    self.all_force_sep_matrices_before_filter[i][j], current_indices)
                self.all_force_time_matrices_after_filter[i][j] = select_double_columns(
                    self.all_force_time_matrices_before_filter[i][j], current_indices)
        # if not self.debug:
        if not self.debug:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Save report?")
            msg.setText("Do you want to save a filtering report?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
        else:
            response = QMessageBox.No
        if response == QMessageBox.Yes:
            save_xlsx_folder = QFileDialog.getExistingDirectory(self, "Select folder to save filtering report")
            if not save_xlsx_folder:
                return

            filename = self.experiment_name + " filtering report.xlsx"
            xls_file_full_path = save_xlsx_folder + "/" + filename
            sheet_names = [["" for i in range(self.duplicatenum)] for j in range(self.forcenum)]

            for i in range(self.forcenum):
                for j in range(self.duplicatenum):
                    name = str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                     2)) + " " + self.F_power_name + " sample #" + str(j + 1)
                    sheet_name = name
                    headers = ["Taken out:", "Reason:"]
                    string_list_to_save = [list(pair) for pair in zip(self.all_headers_vectors_taken_out[i][j],
                                                                      self.filtered_out_indices_reason[i][j])]
                    if not string_list_to_save:
                        string_list_to_save = [["No data taken out for this instance!", ""]]
                    if i == 0 and j == 0:
                        save_list_to_excel(string_list_to_save, save_xlsx_folder, filename, headers, sheet_name)
                    else:
                        add_sheet_to_existing_excel(xls_file_full_path, sheet_name, string_list_to_save, headers)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
            msg_box.setWindowTitle("Confirmation")
            msg_box.setText("xlsx saved successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()  # Show the popup

    def handle_data_after_single_remove_from_d0_panel(self):
        self.final_trace_number = len(self.all_index_vectors_after_filtering[self.curr_force_i][self.curr_sample_i])
        i = self.curr_force_i
        j = self.curr_sample_i
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                current_indices = self.all_index_vectors_after_filtering[i][j]
                while len(current_indices) > self.final_trace_number:
                    integer_to_delete = random.randint(0, (len(current_indices) - 1))
                    del current_indices[integer_to_delete]
                    self.all_headers_vectors_taken_out[i][j].append(
                        self.all_headers_vectors_before_filter[i][j][integer_to_delete])
                    self.filtered_out_indices_reason[i][j].append("Randomly removed")
                self.all_index_vectors_after_filtering[i][j] = current_indices
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                current_indices = np.array(self.all_index_vectors_after_filtering[i][j])
                self.all_force_sep_matrices_after_filter[i][j] = select_double_columns(
                    self.all_force_sep_matrices_before_filter[i][j], current_indices)
                self.all_force_time_matrices_after_filter[i][j] = select_double_columns(
                    self.all_force_time_matrices_before_filter[i][j], current_indices)
        # if not self.debug:
        if not self.debug:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Save report?")
            msg.setText("Do you want to save a filtering report?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
        else:
            response = QMessageBox.No
        if response == QMessageBox.Yes:
            save_xlsx_folder = QFileDialog.getExistingDirectory(self, "Select folder to save filtering report")
            if not save_xlsx_folder:
                return

            filename = self.experiment_name + " filtering report.xlsx"
            xls_file_full_path = save_xlsx_folder + "/" + filename
            sheet_names = [["" for i in range(self.duplicatenum)] for j in range(self.forcenum)]

            for i in range(self.forcenum):
                for j in range(self.duplicatenum):
                    name = str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                     2)) + " " + self.F_power_name + " sample #" + str(j + 1)
                    sheet_name = name
                    headers = ["Taken out:", "Reason:"]
                    string_list_to_save = [list(pair) for pair in zip(self.all_headers_vectors_taken_out[i][j],
                                                                      self.filtered_out_indices_reason[i][j])]
                    if not string_list_to_save:
                        string_list_to_save = [["No data taken out for this instance!", ""]]
                    if i == 0 and j == 0:
                        save_list_to_excel(string_list_to_save, save_xlsx_folder, filename, headers, sheet_name)
                    else:
                        add_sheet_to_existing_excel(xls_file_full_path, sheet_name, string_list_to_save, headers)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
            msg_box.setWindowTitle("Confirmation")
            msg_box.setText("xlsx saved successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()  # Show the popup


    def plot_on_ai_viewing_graph(self, x_data, y_data, title="Graph", xlabel="X-axis", ylabel="Y-axis"):
        """
        Plots a graph on the Matplotlib canvas named self.AI_Viewing_Graph.

        Parameters:
        - x_data: List or array of X-axis data points.
        - y_data: List or array of Y-axis data points.
        - title: Title of the graph (default: 'Graph').
        - xlabel: Label for the X-axis (default: 'X-axis').
        - ylabel: Label for the Y-axis (default: 'Y-axis').
        """
        # Clear the previous plot
        self.AI_Viewing_Graph.axes.clear()

        # Plot new data
        self.AI_Viewing_Graph.axes.plot(x_data, y_data, marker='o')

        # Set titles and labels
        self.AI_Viewing_Graph.axes.set_title(title)
        self.AI_Viewing_Graph.axes.set_xlabel(xlabel)
        self.AI_Viewing_Graph.axes.set_ylabel(ylabel)

        # Refresh the canvas
        self.AI_Viewing_Graph.draw()

    def force_curve_list_updated(self):
        if isinstance(self.all_ipts, int):
            plot_default_graph_in_find_d0_graph(self)
        else:
            plot_graph_in_find_d0_graph(self)
        self.initiate_slider()

    def force_curve_list_2_updated(self):
        plot_graph_in_contact_graph(self)
        # self.initiate_contact_slider()

    def default_d0_button_clicked(self):
        i = self.force_curve_list.currentRow()
        def_ipt = self.def_all_ipts[i]
        self.all_ipts[i] = def_ipt
        self.initiate_slider()

    def Save_d0_button_clicked(self):
        if not self.debug:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("Are you sure?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
        else:
            response = QMessageBox.Yes
        if response == QMessageBox.Yes:
            self.all_d0_vectors[self.curr_force_i][self.curr_sample_i] = self.d0
            self.all_ipt_vectors[self.curr_force_i][self.curr_sample_i] = self.all_ipts
            self.d0_averages_divided[self.curr_force_i][self.curr_sample_i] = np.mean(np.array(self.d0))
            a_vector = np.zeros(len(self.d0))
            for i in range(len(self.d0)):
                a_vector[i] = calculate_length_scale(self, self.d0[i], self.R_micron)

            self.all_a_vectors[self.curr_force_i][self.curr_sample_i] = a_vector
            self.force_curve_list.setEnabled(False)
            self.force_curve_list.setCurrentRow(0)
            self.calc_d0_button.setEnabled(True)
            self.curr_force_i = self.force_combobox_2.currentIndex()
            self.curr_sample_i = self.sample_combobox_2.currentIndex()
            self.sample_combobox_2.setEnabled(True)
            self.force_combobox_2.setEnabled(True)
            self.update_current_matrices_filtered()
            self.d0_slider.setEnabled(False)
            self.default_d0_button.setEnabled(False)
            self.Save_d0_button.setEnabled(False)
            # self.model_find_d0_combobox.setEnabled(False)
            self.Indentation_depth_label.setEnabled(False)
            self.Change_model_label.setEnabled(False)
            self.show_histogram_button.setEnabled(False)
            self.Lowest_button.setEnabled(False)
            self.Highest_button.setEnabled(False)
            self.remove_current_trace_d0_button.setEnabled(False)


        else:
            return
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                all_a = self.all_a_vectors[i][j]
                if (isinstance(all_a, int)):
                    return
        if not self.debug:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("All indentation depths calculated! Do you want to move on to the next step?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
        else:
            response = QMessageBox.Yes
        if response == QMessageBox.Yes:
            self.FitTab.setCurrentIndex(3)
            self.FitTab.setTabEnabled(2, False)
            self.FitTab.setTabEnabled(3, True)
            self.FitTab.setTabEnabled(4, True)
            # self.FitTab.setTabEnabled(5, True)
            self.merge_all_samples()
            plot_graph_in_analysis_graph(self)
            self.initialize_tab_4()
            # self.initialize_tab_5()
            self.current_tab = 3
            # self.initialize_tab_3()
        else:
            self.force_curve_list.setEnabled(True)
            self.Save_d0_button.setEnabled(False)
            self.force_curve_list.setCurrentRow(0)
            self.calc_d0_button.setEnabled(True)
            self.curr_force_i = self.force_combobox_2.currentIndex()
            self.curr_sample_i = self.sample_combobox_2.currentIndex()
            self.sample_combobox_2.setEnabled(True)
            self.force_combobox_2.setEnabled(True)
            self.update_current_matrices_filtered()
            self.d0_slider.setEnabled(True)
            self.default_d0_button.setEnabled(True)
            self.Save_d0_button.setEnabled(True)
            # self.model_find_d0_combobox.setEnabled(True)
            self.Indentation_depth_label.setEnabled(True)
            self.Change_model_label.setEnabled(True)


            return

    def model_find_d0_combobox_changed(self):
        if (not self.current_model == self.model_find_d0_combobox.currentText()):
            current_index = self.model_find_d0_combobox.currentIndex()
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("This will override any current calculation for this instance, are you sure?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)
            response = msg.exec_()
            if response == QMessageBox.Yes:
                self.calc_d0_button_clicked()
                self.initiate_slider()
                self.current_model_index = current_index
                self.current_model = self.model_find_d0_combobox.currentText()
            else:
                self.model_find_d0_combobox.setCurrentIndex(self.current_model_index)
                return

    def merge_all_samples(self):
        for i in range(self.forcenum):
            self.all_force_sep_matrices_after_filter_merged[i] = []
            self.all_force_time_matrices_after_filter_merged[i] = []
            for j in range(self.duplicatenum):
                curr_all_index_vectors_after_filtering = self.all_index_vectors_after_filtering[i][j]
                curr_all_force_sep_matrices_after_filter = self.all_force_sep_matrices_after_filter[i][j]
                curr_all_force_time_matrices_after_filter = self.all_force_time_matrices_after_filter[i][j]
                curr_all_d0_vectors_merged = self.all_d0_vectors[i][j]
                curr_all_a_vectors_merged = self.all_a_vectors[i][j]
                curr_all_ipt_vectors_merged = self.all_ipt_vectors[i][j]
                # curr_d0_averages_final = self.d0_averages_divided[i][j]
                self.all_index_vectors_after_filtering_merged[i].extend(curr_all_index_vectors_after_filtering)
                for k in curr_all_index_vectors_after_filtering:
                    name = "sample " + str(j + 1) + " trace #" + str(k + 1)
                    self.trace_names_after_filtering_merged_sep[i].extend([name + " sep [m]"])
                    self.trace_names_after_filtering_merged_sep[i].extend([name + " force [N]"])
                    self.trace_names_after_filtering_merged_time[i].extend([name + " time [s]"])
                    self.trace_names_after_filtering_merged_time[i].extend([name + " force [N]"])

                self.all_d0_vectors_merged[i].extend(curr_all_d0_vectors_merged)
                self.all_a_vectors_merged[i].extend(curr_all_a_vectors_merged)
                self.all_ipt_vectors_merged[i].extend(curr_all_ipt_vectors_merged)
                self.all_force_sep_matrices_after_filter_merged[i] = merge_2d_lists(
                    self.all_force_sep_matrices_after_filter_merged[i], curr_all_force_sep_matrices_after_filter)
                self.all_force_time_matrices_after_filter_merged[i] = merge_2d_lists(
                    self.all_force_time_matrices_after_filter_merged[i], curr_all_force_time_matrices_after_filter)
        for i in range(self.forcenum):
            self.d0_averages_final[i] = np.mean(np.array(self.all_d0_vectors_merged[i]))
            self.d0_stds_final[i] = np.std(np.array(self.all_d0_vectors_merged[i]))
        self.d0_averages_final = np.array(self.d0_averages_final)
        self.d0_stds_final = np.array(self.d0_stds_final)

        for i in range(self.forcenum):
            current_FT_matrix = Normalize_norm_2d_list(self.all_force_time_matrices_after_filter_merged[i],
                                                       self.forcevalues_arr[i])
            self.norm_all_matrices[i] = current_FT_matrix
            self.visco_all_matrices[i] = Normalize_force_2d_list(current_FT_matrix)
            self.poro_all_matrices[i] = Normalize_force_2d_list(current_FT_matrix)
            a_vector = np.array(self.all_a_vectors_merged[i])
            self.poro_all_matrices[i] = Normalize_time_2d_list(self.poro_all_matrices[i], a_vector)
            self.norm_x_vectors_avg[i], self.norm_y_vectors_avg[i], self.norm_x_vectors_std[i], self.norm_y_vectors_std[
                i] = Calculate_average_and_std(self.norm_all_matrices[i])
            self.poro_x_vectors_avg[i], self.poro_y_vectors_avg[i], self.poro_x_vectors_std[i], self.poro_y_vectors_std[
                i] = Calculate_average_and_std(self.poro_all_matrices[i])
            self.poro_all_matrices[i] = interp_after_x_interruption(self.poro_all_matrices[i],
                                                                    self.poro_x_vectors_avg[i])
            self.visco_x_vectors_avg[i], self.visco_y_vectors_avg[i], self.visco_x_vectors_std[i], \
            self.visco_y_vectors_std[i] = Calculate_average_and_std(self.visco_all_matrices[i])
            self.norm_y_vectors_UB[i] = self.norm_y_vectors_avg[i] + self.norm_y_vectors_std[i]
            self.visco_y_vectors_UB[i] = self.visco_y_vectors_avg[i] + self.visco_y_vectors_std[i]
            self.poro_y_vectors_UB[i] = self.poro_y_vectors_avg[i] + self.poro_y_vectors_std[i]
            self.norm_y_vectors_LB[i] = self.norm_y_vectors_avg[i] - self.norm_y_vectors_std[i]
            self.visco_y_vectors_LB[i] = self.visco_y_vectors_avg[i] - self.visco_y_vectors_std[i]
            self.poro_y_vectors_LB[i] = self.poro_y_vectors_avg[i] - self.poro_y_vectors_std[i]
        self.create_variance_analysis_vectors()

    def check_type_combobox_changed(self):
        plot_graph_in_analysis_graph(self)

    def show_histogram_button_clicked(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        height = screen_size.height()
        font_size = int((18 / 1440) * height)
        title_size = int((28 / 1440) * height)

        self.Force_Sep_Graph.ax.clear()  # Clear previous plots
        d0_average = np.mean(self.d0)
        d0_std = np.std(self.d0)
        num_bins = Freedman_Diaconis(self.d0)
        length_unit_text = self.R_power_name
        force_unit_text = self.newton_scale_combobox.currentText()
        self.Indentation_depth_label.setText("δ0 = " + str(round(d0_average * 10 ** (-self.R_power), 2)) + " ± " + str(
            round(d0_std * 10 ** (-self.R_power), 2)) + " " + length_unit_text)

        title_text = "Duplicate #" + str(self.curr_sample_i + 1) + ", " + str(
            round(self.forcevalues_arr[self.curr_force_i] * 10 ** (-self.F_power),
                  2)) + " " + force_unit_text + " histogram"

        self.Force_Sep_Graph.ax.hist(self.d0, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
        self.Force_Sep_Graph.ax.set_title(title_text, fontsize=title_size)
        self.Force_Sep_Graph.ax.set_xlabel("Indentation Depth [m]", fontsize=font_size)
        self.Force_Sep_Graph.ax.set_ylabel("Frequency", fontsize=font_size)
        self.Force_Sep_Graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        font_size = int(font_size * (24 / 18))
        self.Force_Sep_Graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
        self.Force_Sep_Graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
        self.Force_Sep_Graph.ax.tick_params(axis='both', labelsize=font_size)
        textbox_text = "N= " + str(self.d0.shape[0])
        self.Force_Sep_Graph.ax.text(0.8, 0.8, textbox_text, fontsize=title_size, color='black',
                                     transform=self.Force_Sep_Graph.ax.transAxes,
                                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

        # self.Force_Sep_Graph.ax.grid(True)
        self.Force_Sep_Graph.fig.tight_layout()

        self.Force_Sep_Graph.draw()

    def Lowest_button_clicked(self):
        min_pt = np.argmin(self.d0)
        if (min_pt == self.force_curve_list.currentRow()):
            plot_graph_in_find_d0_graph(self)
        else:
            self.force_curve_list.setCurrentRow(min_pt)

    def Highest_button_clicked(self):
        max_pt = np.argmax(self.d0)
        if (max_pt == self.force_curve_list.currentRow()):
            plot_graph_in_find_d0_graph(self)
        else:
            self.force_curve_list.setCurrentRow(max_pt)

    def create_variance_analysis_vectors(self):
        all_max_x_values = []
        for i in range(self.forcenum):
            all_max_x_values.append(np.max(np.array(self.poro_x_vectors_avg[i])))
        max_x_value = np.min(np.array(all_max_x_values))
        max_x_value_i = np.argmin(np.array(all_max_x_values))
        final_x_values_for_variance = self.poro_x_vectors_avg[max_x_value_i]
        all_max_args = []

        for i in range(self.forcenum):
            if not np.all(np.diff(self.poro_x_vectors_avg[i]) >= 0):
                raise ValueError("Input array poro_x_vectors_avg[", i, "] must be sorted in ascending order.")
            all_max_args.append(np.searchsorted(self.poro_x_vectors_avg[i], max_x_value, side='right') - 1)
            self.variance_poro_vectors[i] = np.interp(final_x_values_for_variance, self.poro_x_vectors_avg[i],
                                                      self.poro_y_vectors_avg[i])
        variance_poro_vectors_to_calc = np.array(self.variance_poro_vectors)
        variance_visco_vectors_to_calc = np.array(self.visco_y_vectors_avg)
        self.variance_visco_vector = np.var(variance_visco_vectors_to_calc, axis=0)
        self.variance_poro_vector = np.var(variance_poro_vectors_to_calc, axis=0)

    def skip_d0_button_clicked(self):
        self.debug = True
        for i in range(self.duplicatenum):
            if i > 0:
                self.sample_combobox_2.setCurrentIndex(i)
            for j in range(self.forcenum):
                if j > 0:
                    self.force_combobox_2.setCurrentIndex(j)
                self.calc_d0_button_clicked()
                self.Save_d0_button_clicked()
        self.debug = False

    def debug_button_clicked(self):
        self.debug = True
        self.enter_data_btn_pressed()
        #self.experiment_name = "debug"
        self.experiment_name = "BSA in Tris"
        self.force_values_input.setValue(25)
        self.enter_force_nN_btn_pressed()
        self.force_values_input.setValue(90)
        self.enter_force_nN_btn_pressed()
        self.force_values_input.setValue(400)
        self.enter_force_nN_btn_pressed()
        # self.duplicatenum=2
        for i in range(6):
            self.load_new_forcesep_csv()
            self.load_new_forcetime_csv()
            self.next_tab_0_button_pressed()
            self.debug_upload_csv_i += 1
        #self.debug = False
        #self.start_over_AI_button_clicked()
        """
        for i in range(self.forcenum):
            for j in range(self.duplicatenum):
                self.curr_force_i = i
                self.curr_sample_i = j
                self.update_current_matrices()
                if (i + j) > 0:
                    self.force_sep_UL_radio_button.setChecked(True)
                    self.force_time_UL_radio_button.setChecked(False)
                self.elbow_check_clicked()
                self.apply_clustreing_button_clicked()
                self.filtering_accepted_button_clicked()
                self.force_sep_UL_radio_button.setChecked(False)
                self.force_time_UL_radio_button.setChecked(True)
                self.elbow_check_clicked()
                self.apply_clustreing_button_clicked()
                self.filtering_accepted_button_clicked()
        self.skip_d0_button_clicked()
        self.FitTab.setCurrentIndex(4)
        """

    def indenter_type_combobox_changed(self):
        indenter_type_i = self.indenter_type_combobox.currentIndex()
        if indenter_type_i == 0:
            self.R_micron_spinbox.setValue(5.1)
            self.meter_scale_combobox.clear()
            self.meter_scale_combobox.addItem("m")
            self.meter_scale_combobox.addItem("mm")
            self.meter_scale_combobox.addItem("μm")
            self.meter_scale_combobox.addItem("nm")
            self.meter_scale_combobox.setCurrentIndex(2)
            self.Indenter_Label.setText("Indenter Radius:")
        elif indenter_type_i == 1:
            self.R_micron_spinbox.setValue(17)
            self.meter_scale_combobox.clear()
            self.meter_scale_combobox.addItem("°")
            self.meter_scale_combobox.addItem("rad")
            self.meter_scale_combobox.setCurrentIndex(0)
            self.Indenter_Label.setText("Half Angle:")

        else:
            raise ValueError("Incorrect indenter type")

    def save_graphs_button_clicked(self):
        save_graphs_folder = QFileDialog.getExistingDirectory(self, "Select folder to save graphs")



        if save_graphs_folder:
            self.original_size = self.size()
            for i in range(5):
                if i == 0:
                    change_window_size_image_export(self,569, 549)
                else:
                    change_window_size_image_export(self,338,527)
                self.check_type_combobox.setCurrentIndex(i)
                filename = self.experiment_name + " " + self.check_type_combobox.currentText()
                save_graph_as_tiff(self, save_graphs_folder, filename)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
            msg_box.setWindowTitle("Confirmation")
            msg_box.setText("Graphs saved successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()  # Show the popup
        else:
            restore_window_size(self)
            return
        restore_window_size(self)

    def save_experiment_button_clicked(self):
        if self.FitTab.currentIndex() == 0:
            msg_box = QMessageBox()
            msg_box.setWindowTitle("Not enough data")
            msg_box.setText("Not enough data to save.")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()
            return False
        save_exp_folder = QFileDialog.getExistingDirectory(self, "Select folder to save experiment parameters")
        if save_exp_folder:
            save_experiment_params_pickle(self, save_exp_folder,
                                          filename=self.experiment_name + " experiment params V" + str(
                                              int(self.saving_version)) + ".json")
            self.saving_version = self.saving_version + 1
            if self.debug:
                self.FitTab.setTabEnabled(0, True)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
            msg_box.setWindowTitle("Confirmation")
            msg_box.setText("Experiment saved successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()  # Show the popup
            return True
        else:
            return False

    def load_experiment_button_clicked(self):
        load_exp_folder, _ = QFileDialog.getOpenFileName(self, "Open Experiment File", "",
                                                         "JSON Files (*.json)")
        if load_exp_folder:
            params = load_experiment_params(load_exp_folder)
            if "experiment_name_DigiGel" in params:
                self.load_experiment_button.setEnabled(False)
                self.curr_force_sep_data = params["curr_force_sep_data"]
                self.d0 = params["d0"]
                self.all_d0_vectors = params["all_d0_vectors"]
                self.all_a_vectors = params["all_a_vectors"]
                self.d0_default = params["d0_default"]
                self.all_ipts = params["all_ipts"]
                #print("Assigned self.all_ipts at 3831:", self.all_ipts, type(self.all_ipts))
                self.def_all_ipts = params["def_all_ipts"]
                self.all_ipt_vectors = params["all_ipt_vectors"]
                self.all_d0_vectors_merged = params["all_d0_vectors_merged"]
                self.all_a_vectors_merged = params["all_a_vectors_merged"]
                self.all_ipt_vectors_merged = params["all_ipt_vectors_merged"]
                self.all_E_star_vectors_merged = params["all_E_star_vectors_merged"]
                self.forcenum = params["forcenum"]
                self.curr_force_i = params["curr_force_i"]
                self.curr_sample_i = params["curr_sample_i"]
                self.duplicatenum = params["duplicatenum"]
                self.R_micron = params["R_micron"]
                self.n_clusters = params["n_clusters"]
                self.final_trace_number = params["final_trace_number"]
                self.current_model_index = params["current_model_index"]
                self.current_model = params["current_model"]
                self.traceNames = params["traceNames"]
                self.graphtitle_arr = params["graphtitle_arr"]
                self.forcevalues_arr = params["forcevalues_arr"]
                self.all_force_sep_AI_indices = params["all_force_sep_AI_indices"]
                self.all_force_time_AI_indices = params["all_force_time_AI_indices"]
                self.all_force_sep_matrices_before_filter = params["all_force_sep_matrices_before_filter"]
                self.all_force_time_matrices_before_filter = params["all_force_time_matrices_before_filter"]
                self.all_index_vectors_after_filtering = params["all_index_vectors_after_filtering"]
                self.all_force_sep_matrices_after_filter = params["all_force_sep_matrices_after_filter"]
                self.all_force_time_matrices_after_filter = params["all_force_time_matrices_after_filter"]
                self.all_index_vectors_after_filtering_merged = params["all_index_vectors_after_filtering_merged"]
                self.all_force_sep_matrices_after_filter_merged = params["all_force_sep_matrices_after_filter_merged"]
                self.all_force_time_matrices_after_filter_merged = params["all_force_time_matrices_after_filter_merged"]
                self.current_force_sep_matrix = params["current_force_sep_matrix"]
                self.current_force_time_matrix = params["current_force_time_matrix"]
                self.norm_x_vectors_avg = params["norm_x_vectors_avg"]
                self.visco_x_vectors_avg = params["visco_x_vectors_avg"]
                self.poro_x_vectors_avg = params["poro_x_vectors_avg"]
                self.norm_y_vectors_avg = params["norm_y_vectors_avg"]
                self.visco_y_vectors_avg = params["visco_y_vectors_avg"]
                self.poro_y_vectors_avg = params["poro_y_vectors_avg"]
                self.norm_x_vectors_std = params["norm_x_vectors_std"]
                self.visco_x_vectors_std = params["visco_x_vectors_std"]
                self.poro_x_vectors_std = params["poro_x_vectors_std"]
                self.norm_y_vectors_std = params["norm_y_vectors_std"]
                self.visco_y_vectors_std = params["visco_y_vectors_std"]
                self.poro_y_vectors_std = params["poro_y_vectors_std"]
                self.norm_x_vectors_UB = params["norm_x_vectors_UB"]
                self.visco_x_vectors_UB = params["visco_x_vectors_UB"]
                self.poro_x_vectors_UB = params["poro_x_vectors_UB"]
                self.norm_y_vectors_UB = params["norm_y_vectors_UB"]
                self.visco_y_vectors_UB = params["visco_y_vectors_UB"]
                self.poro_y_vectors_UB = params["poro_y_vectors_UB"]
                self.norm_x_vectors_LB = params["norm_x_vectors_LB"]
                self.visco_x_vectors_LB = params["visco_x_vectors_LB"]
                self.poro_x_vectors_LB = params["poro_x_vectors_LB"]
                self.norm_y_vectors_LB = params["norm_y_vectors_LB"]
                self.visco_y_vectors_LB = params["visco_y_vectors_LB"]
                self.poro_y_vectors_LB = params["poro_y_vectors_LB"]
                self.norm_all_matrices = params["norm_all_matrices"]
                self.visco_all_matrices = params["visco_all_matrices"]
                self.poro_all_matrices = params["poro_all_matrices"]
                self.force_sep_uploaded_flag = params["force_sep_uploaded_flag"]
                self.force_time_uploaded_flag = params["force_time_uploaded_flag"]
                # self.current_force_signal = params["current_force_signal"]
                self.current_sep_columns = params["current_sep_columns"]
                self.current_force_columns = params["current_force_columns"]
                self.debug = params["debug"]
                self.current_list_of_list_of_clusters = params["current_list_of_list_of_clusters"]
                self.current_index_vector = params["current_index_vector"]
                self.F_power = params["F_power"]
                self.R_power = params["R_power"]
                self.F_power_name = params["F_power_name"]
                self.R_power_name = params["R_power_name"]
                self.d0_averages_divided = params["d0_averages_divided"]
                self.d0_averages_final = params["d0_averages_final"]
                self.d0_stds_final = params["d0_stds_final"]
                self.current_force_sep_CSV_length = params["current_force_sep_CSV_length"]
                self.current_force_time_CSV_length = params["current_force_time_CSV_length"]
                self.debug_upload_csv_i = params["debug_upload_csv_i"]
                self.variance_visco_vector = params["variance_visco_vector"]
                self.variance_poro_vector = params["variance_poro_vector"]
                self.variance_poro_vectors = params["variance_poro_vectors"]
                self.experiment_name = params["experiment_name_DigiGel"]
                self.conical_indenter = params["conical_indenter"]
                self.spherical_indenter = params["spherical_indenter"]
                self.half_angle_indenter_rad = params["half_angle_indenter_rad"]
                self.tan_alpha_indenter = params["tan_alpha_indenter"]
                self.current_tab = params["current_tab"]
                self.trace_names_after_filtering_merged_sep = params["trace_names_after_filtering_merged_sep"]
                self.trace_names_after_filtering_merged_time = params["trace_names_after_filtering_merged_time"]
                self.D_p = params["D_p"]
                self.G = params["G"]
                self.nu = params["nu"]
                self.beta = params["beta"]
                self.C_beta = params["C_beta"]
                self.D_p_std = params["D_p_std"]
                self.G_std = params["G_std"]
                self.nu_std = params["nu_std"]
                self.beta_std = params["beta_std"]
                self.C_beta_std = params["C_beta_std"]
                self.D_p_tot = params["D_p_tot"]
                self.G_tot = params["G_tot"]
                self.nu_tot = params["nu_tot"]
                self.beta_tot = params["beta_tot"]
                self.C_beta_tot = params["C_beta_tot"]
                self.D_p_std_tot = params["D_p_std_tot"]
                self.G_std_tot = params["G_std_tot"]
                self.nu_std_tot = params["nu_std_tot"]
                self.beta_std_tot = params["beta_std_tot"]
                self.C_beta_std_tot = params["C_beta_std_tot"]
                self.r_squared_visco = params["r_squared_visco"]
                self.r_squared_poro = params["r_squared_poro"]
                self.fit_type = params["fit_type"]
                self.visco_y_vectors_avg_fits = params["visco_y_vectors_avg_fits"]
                self.poro_y_vectors_avg_fits = params["poro_y_vectors_avg_fits"]
                self.visco_y_vectors_UB_fits = params["visco_y_vectors_UB_fits"]
                self.poro_y_vectors_UB_fits = params["poro_y_vectors_UB_fits"]
                self.visco_y_vectors_LB_fits = params["visco_y_vectors_LB_fits"]
                self.poro_y_vectors_LB_fits = params["poro_y_vectors_LB_fits"]
                self.all_G_values = params["all_G_values"]
                self.all_nu_values = params["all_nu_values"]
                self.poro_fits_performed_already = params["poro_fits_performed_already"]
                self.visco_fits_performed_already = params["visco_fits_performed_already"]
                self.r_squared_visco_values = params["r_squared_visco_values"]
                self.r_squared_poro_values = params["r_squared_poro_values"]
                self.saving_version = params["saving_version"]
                self.all_headers_vectors_before_filter = params["all_headers_vectors_before_filter"]
                self.all_headers_vectors_taken_out = params["all_headers_vectors_taken_out"]
                self.filtered_out_indices = params["filtered_out_indices"]
                self.filtered_out_indices_reason = params["filtered_out_indices_reason"]

                if self.current_tab == 0:
                    return
                elif self.current_tab > 0:
                    self.max_sep_val, self.min_sep_val = get_even_column_extrema(self.all_force_sep_matrices_before_filter)
                    self.load_ForceSep_csv_button.setEnabled(False)
                    self.load_ForceTime_csv_button.setEnabled(False)
                    self.FitTab.setCurrentIndex(1)
                    self.FitTab.setTabEnabled(1, True)
                    self.FitTab.setTabEnabled(0, False)
                    current_data_matrix = self.all_force_time_matrices_before_filter[self.curr_force_i][
                        self.curr_sample_i]
                    current_y_vector = [row[1] for row in current_data_matrix]
                    current_x_vector = [row[0] for row in current_data_matrix]
                    plot_graph_in_ai_viewing_graph(self, current_x_vector, current_y_vector, True)
                    self.save_experiment_button.setEnabled(True)
                    self.initialize_tab_1()
                    if self.current_tab > 1:
                        self.FitTab.setCurrentIndex(2)
                        self.FitTab.setTabEnabled(1, False)
                        # if self.debug:
                        # self.handle_data_after_AI_filter()
                        self.initialize_tab_2()
                        self.d0 = self.all_d0_vectors[self.curr_force_i][self.curr_sample_i]
                        self.all_ipts = self.all_ipt_vectors[self.curr_force_i][self.curr_sample_i]
                        #print("Assigned self.all_ipts at 3982:", self.all_ipts, type(self.all_ipts))
                        if not self.d0 is None:
                            self.Save_d0_button.setEnabled(True)

                        if self.current_tab > 2:
                            self.FitTab.setCurrentIndex(3)
                            self.initialize_tab_4()
                            self.FitTab.setTabEnabled(4, True)
                            self.FitTab.setTabEnabled(3, True)
                            self.FitTab.setTabEnabled(2, False)
                            self.FitTab.setTabEnabled(1, False)
                            self.check_type_combobox_changed()
                            if self.visco_fits_performed_already or self.poro_fits_performed_already:
                                self.save_fit_graph_button.setEnabled(True)
                                self.save_fit_values_button.setEnabled(True)






            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Error")
                msg.setInformativeText('File does not match this software')
                msg.setWindowTitle("Error")
                font = QFont()
                font.setPointSize(14)  # adjust as needed
                msg.setFont(font)

                msg.exec_()
                return

    def save_filtered_data_button_clicked(self):
        save_xlsx_folder = QFileDialog.getExistingDirectory(self, "Select folder to save experiment parameters")
        if not save_xlsx_folder:
            return

        filename = self.experiment_name + " filtered data.xlsx"
        if not filename:
            return
        xls_file_full_path = save_xlsx_folder + "/" + filename

        for i in range(self.forcenum):
            matrix_to_save = self.all_force_sep_matrices_after_filter_merged[i]
            headers = self.trace_names_after_filtering_merged_sep[i]
            sheetname = "Force vs. Sep, " + str(
                round(self.forcevalues_arr[i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name
            filename = self.experiment_name + " " + sheetname + ".csv"
            # if i==0:
            save_ndarray_to_csv(matrix_to_save, save_xlsx_folder, filename, headers,
                                sheetname)
            # else:
            #    add_sheet_to_existing_excel(xls_file_full_path, sheetname, matrix_to_save, headers)
            matrix_to_save = self.all_force_time_matrices_after_filter_merged[i]
            headers = self.trace_names_after_filtering_merged_time[i]
            sheetname = "Force vs. Time, " + str(
                round(self.forcevalues_arr[i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name
            filename = self.experiment_name + " " + sheetname + ".csv"
            save_ndarray_to_csv(matrix_to_save, save_xlsx_folder, filename, headers,
                                sheetname)

            # add_sheet_to_existing_excel(xls_file_full_path, sheetname, matrix_to_save, headers)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("csv saved successfully!")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()  # Show the popup

    def save_analyzed_data_button_clicked(self):
        save_xlsx_folder = QFileDialog.getExistingDirectory(self, "Select folder to save experiment parameters")
        if not save_xlsx_folder:
            return

        filename = self.experiment_name + " analyzed data.xlsx"
        xls_file_full_path = save_xlsx_folder + "/" + filename
        forces_headers = ["" for i in range(self.forcenum)]
        for i in range(self.forcenum):
            name = str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name
            forces_headers[i] = name

        d0_transposed = np.array(list(map(list, zip(*self.all_d0_vectors_merged))))

        save_ndarray_to_excel(d0_transposed, save_xlsx_folder, filename, forces_headers, "Indentation Depths (m)")
        a_transposed = np.array(list(map(list, zip(*self.all_a_vectors_merged))))
        add_sheet_to_existing_excel(xls_file_full_path, "Length Scales (m)", a_transposed.tolist(), forces_headers)
        d0_averages = np.array(self.d0_averages_final)
        d0_stds = np.array(self.d0_stds_final)
        d0_avg_std_mat = np.vstack((d0_averages, d0_stds))
        row_names = [["d0 averages:"], ["d0 st.dev.:"]]
        d0_avg_std_mat = [row1 + row2 for row1, row2 in zip(row_names, d0_avg_std_mat.tolist())]
        d0_avg_std_headers = [""] + forces_headers
        add_sheet_to_existing_excel(xls_file_full_path, "Average d0", d0_avg_std_mat, d0_avg_std_headers)
        norm_matrix = []
        visco_matrix = []
        poro_matrix = []
        norm_matrix_names = []
        visco_matrix_names = []
        poro_matrix_names = []
        for i in range(self.forcenum):
            norm_matrix.append(self.norm_x_vectors_avg[i])
            norm_matrix.append(self.norm_x_vectors_std[i])
            norm_matrix.append(self.norm_y_vectors_avg[i])
            norm_matrix.append(self.norm_y_vectors_std[i])
            norm_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Norm_X_AVG")
            norm_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Norm_X_STD")
            norm_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Norm_Y_AVG")
            norm_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Norm_Y_STD")
            visco_matrix.append(self.visco_x_vectors_avg[i])
            visco_matrix.append(self.visco_x_vectors_std[i])
            visco_matrix.append(self.visco_y_vectors_avg[i])
            visco_matrix.append(self.visco_y_vectors_std[i])
            visco_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                                2)) + " " + self.F_power_name + " Visco_X_AVG")
            visco_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                                2)) + " " + self.F_power_name + " Visco_X_STD")
            visco_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                                2)) + " " + self.F_power_name + " Visco_Y_AVG")
            visco_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                                2)) + " " + self.F_power_name + " Visco_Y_STD")
            poro_matrix.append(self.poro_x_vectors_avg[i])
            poro_matrix.append(self.poro_x_vectors_std[i])
            poro_matrix.append(self.poro_y_vectors_avg[i])
            poro_matrix.append(self.poro_y_vectors_std[i])
            poro_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Poro_X_AVG")
            poro_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Poro_X_STD")
            poro_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Poro_Y_AVG")
            poro_matrix_names.append(str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power),
                                               2)) + " " + self.F_power_name + " Poro_Y_STD")

        variance_matrix = []
        variance_matrix_names = []
        variance_matrix.append(self.variance_visco_vector)
        variance_matrix.append(self.variance_poro_vector)
        variance_matrix_names.append(self.experiment_name + " visco variance")
        variance_matrix_names.append(self.experiment_name + " poro variance")

        norm_matrix = list(map(list, zip(*norm_matrix)))
        visco_matrix = list(map(list, zip(*visco_matrix)))
        poro_matrix = list(map(list, zip(*poro_matrix)))
        variance_matrix = list(map(list, zip(*variance_matrix)))

        add_sheet_to_existing_excel(xls_file_full_path, "Relaxation Graphs Vectors", norm_matrix, norm_matrix_names)
        add_sheet_to_existing_excel(xls_file_full_path, "Visco Test Vectors", visco_matrix, visco_matrix_names)
        add_sheet_to_existing_excel(xls_file_full_path, "Poro Test Vectors", poro_matrix, poro_matrix_names)
        add_sheet_to_existing_excel(xls_file_full_path, "Variance Analysis", variance_matrix, variance_matrix_names)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("xlsx saved successfully!")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()  # Show the popup

    def initialize_tab_4(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        height = screen_size.height()
        font_size = int((18 / 1440) * height)
        title_size = int((24 / 1440) * height)

        for i in range(self.forcenum):
            self.force_fit_combobox.addItem(
                str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name)
        check_type = 1
        i = 0
        self.fit_graph.ax.clear()  # Clear previous plots
        d0_value = self.d0_averages_final[i]
        d0_std = self.d0_stds_final[i]
        length_unit_text = self.R_power_name
        labeltext = "$δ_{0}$ = " + str(round(d0_value * 10 ** (-self.R_power), 2)) + " ± " + str(
            round(d0_std * 10 ** (-self.R_power), 2)) + " " + length_unit_text

        x = self.visco_x_vectors_avg[i]
        y = self.visco_y_vectors_avg[i]
        error = self.visco_y_vectors_std[i]
        x = np.array(x)
        y = np.array(y)
        error = np.array(error)
        self.fit_graph.ax.plot(x, y, marker="o", linestyle="None", linewidth=3, label=labeltext)
        self.fit_graph.ax.fill_between(x, y - error, y + error, alpha=0.2)
        self.fit_graph.ax.set_title("Viscoelasticity fit - " + self.experiment_name, fontsize=title_size)
        self.fit_graph.ax.set_xlabel("Time [s]", fontsize=font_size)
        self.fit_graph.ax.set_ylabel("Normalized force", fontsize=font_size)
        self.fit_graph.ax.legend(loc="upper right", fontsize=title_size)
        self.fit_graph.ax.ticklabel_format(axis='both', style='sci', scilimits=(-2, 2))
        title_size = int(title_size * (24 / 28))
        self.fit_graph.ax.yaxis.get_offset_text().set_fontsize(font_size)
        self.fit_graph.ax.xaxis.get_offset_text().set_fontsize(font_size)
        self.fit_graph.ax.tick_params(axis='both', labelsize=font_size)
        self.fit_graph.ax.grid(True)
        self.fit_graph.draw()

    def make_poro_fits_button_clicked(self):
        self.fit_for_current_force_label.setVisible(True)
        self.fit_for_all_force_label.setVisible(True)
        self.current_parameter_label_1.setVisible(True)
        self.current_parameter_label_2.setVisible(True)
        self.current_parameter_label_3.setVisible(True)
        self.current_parameter_label_4.setVisible(True)
        self.parameter_label_1.setVisible(True)
        self.parameter_label_2.setVisible(True)
        self.parameter_label_3.setVisible(True)
        self.parameter_label_4.setVisible(True)


        self.fit_type = "poro"
        self.force_fit_combobox.setEnabled(True)
        self.poro_fits_performed_already = False
        if not self.poro_fits_performed_already:
            self.poro_fits_performed_already = True
            self.r_squared_poro = 1
            for i in range(self.forcenum):
                x = self.poro_x_vectors_avg[i]
                y = self.poro_y_vectors_avg[i]
                x_data = np.array(x)
                y_data = np.array(y)
                # Perform the curve fitting
                if (self.conical_indenter):
                    params_avg, covariance_avg = curve_fit(poro_cone, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_avg_fits[i] = poro_cone(x_data, *params_avg)
                else:
                    params_avg, covariance_avg = curve_fit(poro_sphere, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_avg_fits[i] = poro_sphere(x_data, *params_avg)
                if np.isnan(params_avg).any():
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Calculation Error")
                    msg.setText("Statistical fit failed. Check your data.")
                    font = QFont()
                    font.setPointSize(14)  # adjust as needed
                    msg.setFont(font)

                    msg.exec_()
                    return

                residuals = self.poro_y_vectors_avg[i] - self.poro_y_vectors_avg_fits[i]
                ss_res = np.sum(residuals ** 2)  # Residual sum of squares
                ss_tot = np.sum((self.poro_y_vectors_avg[i] - np.mean(
                    self.poro_y_vectors_avg[i])) ** 2)  # Total sum of squares
                r_squared = 1 - (ss_res / ss_tot)  # R² value
                self.r_squared_poro_values[i] = r_squared
                if r_squared < self.r_squared_poro:
                    self.r_squared_poro = r_squared
                # x = self.poro_x_vectors_LB[i]
                y = self.poro_y_vectors_LB[i]
                x_data = np.array(x)
                y_data = np.array(y)

                if (self.conical_indenter):
                    params_LB, covariance_LB = curve_fit(poro_cone, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_LB_fits[i] = poro_cone(x_data, *params_LB)
                else:
                    params_LB, covariance_LB = curve_fit(poro_sphere, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_LB_fits[i] = poro_sphere(x_data, *params_LB)

                # x = self.poro_x_vectors_UB[i]
                y = self.poro_y_vectors_UB[i]
                x_data = np.array(x)
                y_data = np.array(y)

                if (self.conical_indenter):
                    params_UB, covariance_UB = curve_fit(poro_cone, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_UB_fits[i] = poro_cone(x_data, *params_UB)
                else:
                    params_UB, covariance_UB = curve_fit(poro_sphere, x_data, y_data, p0=[10 ** -12])
                    self.poro_y_vectors_UB_fits[i] = poro_sphere(x_data, *params_UB)
                self.D_p[i] = 0.5 * (params_UB[0] + params_LB[0])
                self.D_p_std[i] = np.abs(0.5 * (params_UB[0] - params_LB[0]))
                for j in range(self.final_trace_number * self.duplicatenum):
                    currentvector = np.array(self.norm_all_matrices[i])
                    currentvector = currentvector[:, 2 * j + 1]
                    F0 = currentvector[0]
                    #calculate F_end from 5% percent of last values
                    percent = 5
                    n = max(1, int(len(currentvector) * percent / 100))  # make sure at least 1 element
                    end_values = currentvector[-n:]
                    F_end = np.mean(end_values)
                    #F_end = currentvector[-1]
                    if self.conical_indenter:
                        self.all_G_values[i].append(
                            F0 / (4 * self.all_d0_vectors_merged[i][j] * self.all_a_vectors_merged[i][j]))
                    else:
                        self.all_G_values[i].append(
                            F0 / ((16 / 3) * self.all_d0_vectors_merged[i][j] * self.all_a_vectors_merged[i][j]))
                    self.all_nu_values[i].append(1 - 0.5 * (F0 / F_end))
                self.G[i] = np.mean(self.all_G_values[i])
                self.G_std[i] = np.std(self.all_G_values[i])
                self.nu[i] = np.mean(self.all_nu_values[i])
                #LookHere
                #self.nu[i] = 1 - 0.5 * (self.norm_y_vectors_avg[i][0] / self.norm_y_vectors_avg[i][-1])
                self.nu_std[i] = np.std(self.all_nu_values[i])
            self.D_p_tot = np.mean(self.D_p)
            self.D_p_std_tot = np.sqrt(np.sum((np.std(self.D_p_std) ** 2) / self.forcenum))
            self.G_tot = np.mean(self.G)
            self.G_std_tot = np.sqrt(np.sum((np.std(self.G_std) ** 2) / self.forcenum))
            self.nu_tot = np.mean(self.nu)
            self.nu_std_tot = np.sqrt(np.sum((np.std(self.nu_std) ** 2) / self.forcenum))
        self.parameter_label_1.setText("Dp = " + str(round(self.D_p_tot * 10 ** (12), 1)) + " ± " + str(
            round(self.D_p_std_tot * 10 ** (12), 1)) + " μm<sup>2</sup>/s")
        self.parameter_label_2.setText(
            "G = " + str(round(self.G_tot / 1000, 2)) + " ± " + str(round(self.G_std_tot / 1000, 2)) + " kPa")
        self.parameter_label_3.setText("ν = " + str(round(self.nu_tot, 4)) + " ± " + str(round(self.nu_std_tot, 4)))
        self.parameter_label_3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.parameter_label_4.setText("R<sup>2</sup> = " + str(round(self.r_squared_poro, 4)))
        self.current_parameter_label_1.setText("Dp = " + str(round(self.D_p[self.force_fit_combobox.currentIndex()] * 10 ** (12), 1)) + " ± " + str(
            round(self.D_p_std[self.force_fit_combobox.currentIndex()] * 10 ** (12), 1)) + " μm<sup>2</sup>/s")
        self.current_parameter_label_2.setText(
            "G = " + str(round(self.G[self.force_fit_combobox.currentIndex()] / 1000, 2)) + " ± " + str(round(self.G_std[self.force_fit_combobox.currentIndex()] / 1000, 2)) + " kPa")
        self.current_parameter_label_3.setText("ν = " + str(round(self.nu[self.force_fit_combobox.currentIndex()], 4)) + " ± " + str(round(self.nu_std[self.force_fit_combobox.currentIndex()], 4)))
        self.current_parameter_label_3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.current_parameter_label_4.setText("R<sup>2</sup> = " + str(round(self.r_squared_poro_values[self.force_fit_combobox.currentIndex()], 4)))
        self.force_fit_combobox.setCurrentIndex(0)

        plot_graph_in_fit_graph(self)
        self.save_fit_graph_button.setEnabled(True)
        self.save_fit_values_button.setEnabled(True)

    def make_visco_fits_button_clicked(self):
        self.fit_for_current_force_label.setVisible(True)
        self.fit_for_all_force_label.setVisible(True)
        self.current_parameter_label_1.setVisible(True)
        self.current_parameter_label_2.setVisible(True)
        self.current_parameter_label_3.setVisible(True)
        self.current_parameter_label_4.setVisible(True)
        self.parameter_label_1.setVisible(True)
        self.parameter_label_2.setVisible(True)
        self.parameter_label_3.setVisible(True)
        self.parameter_label_4.setVisible(True)

        self.fit_type = "visco"
        self.force_fit_combobox.setEnabled(True)
        self.visco_fits_performed_already = False
        if not self.visco_fits_performed_already:
            self.visco_fits_performed_already = True
            self.r_squared_visco = 1
            for i in range(self.forcenum):
                x = self.norm_x_vectors_avg[i]
                y = self.norm_y_vectors_avg[i]
                y_std = self.norm_y_vectors_std[i]
                x_data = np.array(x, dtype=np.float64)
                y_data = np.array(y, dtype=np.float64)
                Y_DBG = y_data*10**9
                #slope, intercept, r_value, _, _ = linregress(np.log(x_data[2:]), np.log(y_data[2:]))
                #r_squared = r_value**2
                #params_avg = [-slope, np.exp(intercept) * gamma(1 + slope)]
                params_avg, covariance_avg = curve_fit(visco,x_data[2:], y_data[2:],p0=[0.1, y_data[2]], jac=visco_jac, method='lm')
                if np.isnan(params_avg).any():
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setWindowTitle("Calculation Error")
                    msg.setText("Statistical fit failed. Check your data.")
                    font = QFont()
                    font.setPointSize(14)  # adjust as needed
                    msg.setFont(font)

                    msg.exec_()
                    return
                self.visco_y_vectors_avg_fits[i] = visco(x_data, *params_avg)
                residuals = self.norm_y_vectors_avg[i][2:] - self.visco_y_vectors_avg_fits[i][2:]
                ss_res = np.sum(residuals ** 2)  # Residual sum of squares
                ss_tot = np.sum((self.visco_y_vectors_avg_fits[i][2:] - np.mean(
                    self.norm_y_vectors_avg[i][2:])) ** 2)  # Total sum of squares

                r_squared = 1 - (ss_res / ss_tot)  # R² value
                self.r_squared_visco_values[i] = r_squared
                if r_squared < self.r_squared_visco:
                    self.r_squared_visco = r_squared

                y = self.norm_y_vectors_LB[i]
                y_data = np.array(y, dtype=np.float64)
                slope, intercept, r_value, _, _ = linregress(np.log(x_data[2:]), np.log(y_data[2:]))
                #params_LB = [-slope, np.exp(intercept) * gamma(1 + slope)]
                params_LB, covariance_LB = curve_fit(visco,x_data[2:],y_data[2:], p0=[0.1, y_data[2]], jac=visco_jac, method='lm')
                self.visco_y_vectors_LB_fits[i] = visco(x_data, *params_LB)
                y = self.norm_y_vectors_UB[i]

                y_data = np.array(y, dtype=np.float64)
                slope, intercept, r_value, _, _ = linregress(np.log(x_data[2:]), np.log(y_data[2:]))
                #params_UB = [-slope, np.exp(intercept) * gamma(1 + slope)]
                params_UB, covariance_UB = curve_fit(visco,x_data[2:],y_data[2:], p0=[0.1, y_data[2]], jac=visco_jac, method='lm')
                self.visco_y_vectors_UB_fits[i] = visco(x_data, *params_UB)
                self.beta[i] = 0.5 * (params_UB[0] + params_LB[0])
                self.beta_std[i] = np.abs(0.5 * (params_UB[0] - params_LB[0]))
                AEC_middle = 0.5 * (params_UB[1] + params_LB[1])
                AEC_diff = np.abs(0.5 * (params_UB[1] - params_LB[1]))
                a_middle = np.mean(self.all_a_vectors_merged[i])
                a_diff = np.std(self.all_a_vectors_merged[i])
                R = self.R_micron
                self.C_beta[i] = AEC_middle * R / (np.pi * (a_middle ** 3))
                self.C_beta_std[i] = np.sqrt((AEC_diff * R / (np.pi * (a_middle ** 3))) ** 2 + (
                        3 * a_diff * AEC_middle * R / (np.pi * (a_middle ** 4))) ** 2)
            self.beta_tot = np.mean(self.beta)
            self.beta_std_tot = np.sqrt(np.sum((np.std(self.beta_std) ** 2) / self.forcenum))
            self.C_beta_tot = np.mean(self.C_beta)
            self.C_beta_std_tot = np.sqrt(np.sum((np.std(self.C_beta_std) ** 2) / self.forcenum))
        self.parameter_label_1.setText("β = " + str(round(self.beta_tot, 5)) + " ± " + str(round(self.beta_std_tot, 5)))
        self.parameter_label_2.setText(
            "C<sub>β</sub> = " + str(round(self.C_beta_tot, 1)) + " ± " + str(round(self.C_beta_std_tot, 1)) + " Pa*s<sup>" + str(
                round(self.beta_tot, 2))+"</sup>")
        self.parameter_label_3.setText("")
        self.parameter_label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.parameter_label_4.setText("R<sup>2</sup> = " + str(round(self.r_squared_visco, 4)))
        self.current_parameter_label_1.setText("β = " + str(round(self.beta[self.force_fit_combobox.currentIndex()], 3)) + " ± " + str(round(self.beta_std[self.force_fit_combobox.currentIndex()], 3)))
        self.current_parameter_label_2.setText(
            "C<sub>β</sub> = " + str(round(self.C_beta[self.force_fit_combobox.currentIndex()], 1)) + " ± " + str(round(self.C_beta_std[self.force_fit_combobox.currentIndex()], 1)) + " Pa*s<sup>" + str(
                round(self.beta[self.force_fit_combobox.currentIndex()], 2))+"</sup>")
        self.current_parameter_label_3.setText("")
        self.current_parameter_label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.current_parameter_label_4.setText("R<sup>2</sup> = " + str(round(self.r_squared_visco_values[self.force_fit_combobox.currentIndex()], 4)))
        self.force_fit_combobox.setCurrentIndex(0)
        plot_graph_in_fit_graph(self)
        self.save_fit_graph_button.setEnabled(True)
        self.save_fit_values_button.setEnabled(True)

    """def make_visco_fits_button_clicked(self):
        self.fit_for_current_force_label.setVisible(True)
        self.fit_for_all_force_label.setVisible(True)
        self.current_parameter_label_1.setVisible(True)
        self.current_parameter_label_2.setVisible(True)
        self.current_parameter_label_3.setVisible(True)
        self.current_parameter_label_4.setVisible(True)
        self.parameter_label_1.setVisible(True)
        self.parameter_label_2.setVisible(True)
        self.parameter_label_3.setVisible(True)
        self.parameter_label_4.setVisible(True)

        self.fit_type = "visco"
        self.force_fit_combobox.setEnabled(True)

        if not self.visco_fits_performed_already:
            self.visco_fits_performed_already = True
            self.r_squared_visco = 1

            for i in range(self.forcenum):
                x_data = np.array(self.norm_x_vectors_avg[i])
                y_avg = np.array(self.norm_y_vectors_avg[i])
                y_std = np.array(self.norm_y_vectors_std[i])
                y_LB = np.array(self.norm_y_vectors_LB[i])
                y_UB = np.array(self.norm_y_vectors_UB[i])

                # Weighted fit for average data
                try:
                    params_avg, cov_avg = curve_fit(
                        visco,
                        x_data[2:], y_avg[2:],
                        p0=[0.1, y_avg[2]],
                        bounds=([0, 0], [1, np.inf]),
                        sigma=y_std[2:],
                        absolute_sigma=True,
                        method='trf',
                        max_nfev=10000
                    )
                except Exception:
                    self.show_fit_error()
                    return

                self.visco_y_vectors_avg_fits[i] = visco(x_data, *params_avg)

                # R² from linear regression in log-log space (informal check)
                _, _, r_value, _, _ = linregress(np.log(x_data[2:]), np.log(y_avg[2:]))
                r_squared = r_value ** 2
                self.r_squared_visco_values[i] = r_squared
                self.r_squared_visco = min(self.r_squared_visco, r_squared)

                # Unweighted LB fit
                try:
                    params_LB, _ = curve_fit(
                        visco,
                        x_data[2:], y_LB[2:],
                        p0=[0.1, y_LB[2]],
                        bounds=([0, 0], [1, np.inf]),
                        method='trf',
                        max_nfev=10000
                    )
                except Exception:
                    self.show_fit_error()
                    return
                self.visco_y_vectors_LB_fits[i] = visco(x_data, *params_LB)

                # Unweighted UB fit
                try:
                    params_UB, _ = curve_fit(
                        visco,
                        x_data[2:], y_UB[2:],
                        p0=[0.1, y_UB[2]],
                        bounds=([0, 0], [1, np.inf]),
                        method='trf',
                        max_nfev=10000
                    )
                except Exception:
                    self.show_fit_error()
                    return
                self.visco_y_vectors_UB_fits[i] = visco(x_data, *params_UB)

                # Parameter aggregation (mean ± half-difference)
                self.beta[i] = 0.5 * (params_UB[0] + params_LB[0])
                self.beta_std[i] = np.abs(0.5 * (params_UB[0] - params_LB[0]))

                AEC_middle = 0.5 * (params_UB[1] + params_LB[1])
                AEC_diff = np.abs(0.5 * (params_UB[1] - params_LB[1]))

                a_middle = np.mean(self.all_a_vectors_merged[i])
                a_diff = np.std(self.all_a_vectors_merged[i])
                R = self.R_micron

                self.C_beta[i] = AEC_middle * R / (np.pi * a_middle ** 3)
                self.C_beta_std[i] = np.sqrt(
                    (AEC_diff * R / (np.pi * a_middle ** 3)) ** 2 +
                    (3 * a_diff * AEC_middle * R / (np.pi * a_middle ** 4)) ** 2
                )

            # Global statistics
            self.beta_tot = np.mean(self.beta)
            self.beta_std_tot = np.sqrt(np.sum((np.std(self.beta_std) ** 2) / self.forcenum))
            self.C_beta_tot = np.mean(self.C_beta)
            self.C_beta_std_tot = np.sqrt(np.sum((np.std(self.C_beta_std) ** 2) / self.forcenum))

        # UI updates
        self.parameter_label_1.setText(f"β = {self.beta_tot:.3f} ± {self.beta_std_tot:.3f}")
        self.parameter_label_2.setText(
            f"C<sub>β</sub> = {self.C_beta_tot:.1f} ± {self.C_beta_std_tot:.1f} Pa·s<sup>{self.beta_tot:.2f}</sup>"
        )
        self.parameter_label_3.setText("")
        self.parameter_label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.parameter_label_4.setText(f"R<sup>2</sup> = {self.r_squared_visco:.4f}")

        curr_idx = self.force_fit_combobox.currentIndex()
        self.current_parameter_label_1.setText(f"β = {self.beta[curr_idx]:.3f} ± {self.beta_std[curr_idx]:.3f}")
        self.current_parameter_label_2.setText(
            f"C<sub>β</sub> = {self.C_beta[curr_idx]:.1f} ± {self.C_beta_std[curr_idx]:.1f} Pa·s<sup>{self.beta[curr_idx]:.2f}</sup>"
        )
        self.current_parameter_label_3.setText("")
        self.current_parameter_label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.current_parameter_label_4.setText(f"R<sup>2</sup> = {self.r_squared_visco_values[curr_idx]:.4f}")

        self.force_fit_combobox.setCurrentIndex(0)
        plot_graph_in_fit_graph(self)
        self.save_fit_graph_button.setEnabled(True)
        self.save_fit_values_button.setEnabled(True)

    def show_fit_error(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Calculation Error")
        msg.setText("Statistical fit failed. Check your data.")
        font = QFont()
        font.setPointSize(14)
        msg.setFont(font)
        msg.exec_()"""

    def force_fit_combobox_changed(self):
        plot_graph_in_fit_graph(self)
        if self.fit_type == "visco":
            self.current_parameter_label_1.setText(
                "β = " + str(round(self.beta[self.force_fit_combobox.currentIndex()], 3)) + " ± " + str(
                    round(self.beta_std[self.force_fit_combobox.currentIndex()], 3)))
            self.current_parameter_label_2.setText(
                "C<sub>β</sub> = " + str(round(self.C_beta[self.force_fit_combobox.currentIndex()], 1)) + " ± " + str(
                    round(self.C_beta_std[self.force_fit_combobox.currentIndex()], 1)) + " Pa*s<sup>" + str(
                    round(self.beta[self.force_fit_combobox.currentIndex()], 2)) + "</sup>")
            self.current_parameter_label_3.setText("")
            self.current_parameter_label_3.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            self.current_parameter_label_4.setText(
                "R<sup>2</sup> = " + str(round(self.r_squared_visco_values[self.force_fit_combobox.currentIndex()], 4)))
        elif self.fit_type == "poro":
            self.current_parameter_label_1.setText(
                "Dp = " + str(round(self.D_p[self.force_fit_combobox.currentIndex()] * 10 ** (12), 1)) + " ± " + str(
                    round(self.D_p_std[self.force_fit_combobox.currentIndex()] * 10 ** (12), 1)) + " μm<sup>2</sup>/s")
            self.current_parameter_label_2.setText(
                "G = " + str(round(self.G[self.force_fit_combobox.currentIndex()] / 1000, 2)) + " ± " + str(
                    round(self.G_std[self.force_fit_combobox.currentIndex()] / 1000, 2)) + " kPa")
            self.current_parameter_label_3.setText(
                "ν = " + str(round(self.nu[self.force_fit_combobox.currentIndex()], 4)) + " ± " + str(
                    round(self.nu_std[self.force_fit_combobox.currentIndex()], 4)))
            self.current_parameter_label_3.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.current_parameter_label_4.setText(
                "R<sup>2</sup> = " + str(round(self.r_squared_poro_values[self.force_fit_combobox.currentIndex()], 4)))

    def save_fit_graph_button_clicked(self):
        save_graphs_folder = QFileDialog.getExistingDirectory(self, "Select folder to save graphs")
        if not save_graphs_folder:
            return
        if save_graphs_folder:
            for i in range(self.forcenum):
                self.force_fit_combobox.setCurrentIndex(i)
                filename = self.experiment_name + " " + self.force_fit_combobox.currentText() + " fit graph"
                if self.fit_type == "visco":
                    filename += " fractional viscoelastic model"
                elif self.fit_type == "poro":
                    if self.conical_indenter:
                        filename += " poroelastic model, conical indenter"
                    else:
                        filename += " poroelastic model, spherical indenter"

                save_graph_as_tiff(self, save_graphs_folder, filename)
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
            msg_box.setWindowTitle("Confirmation")
            msg_box.setText("Graphs saved successfully!")
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec_()  # Show the popup
        else:
            return

    def save_fit_values_button_clicked(self):
        save_xlsx_folder = QFileDialog.getExistingDirectory(self, "Select folder to save experiment parameters")
        if not save_xlsx_folder:
            return
        if self.fit_type == "visco":
            model_name = " fractional viscoelastic"
        elif self.fit_type == "poro":
            if self.conical_indenter:
                model_name = " poroelastic model, conical indenter"
            else:
                model_name = " poroelastic model, spherical indenter"

        filename = self.experiment_name + model_name + " model parameters.xlsx"
        xls_file_full_path = save_xlsx_folder + "/" + filename
        forces_headers = ["" for i in range(self.forcenum + 2)]
        for i in range(self.forcenum):
            name = str(round(self.forcevalues_arr[i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name
            forces_headers[i + 1] = name
        forces_headers[self.forcenum + 1] = "Total"
        if self.fit_type == "visco":
            row_names = [["β:"], ["Δβ:"], ["Cβ [Pa*s^" + str(round(self.beta_tot, 2)) + "]:"],
                         ["ΔCβ [Pa*s^" + str(round(self.beta_tot, 2)) + "]:"], ["R^2:"]]
            data_to_save = [[0 for i in range(self.forcenum + 1)] for j in range(5)]
            for i in range(self.forcenum):
                data_to_save[0][i] = self.beta[i]
                data_to_save[1][i] = self.beta_std[i]
                data_to_save[2][i] = self.C_beta[i]
                data_to_save[3][i] = self.C_beta_std[i]
                data_to_save[4][i] = self.r_squared_visco_values[i]
            data_to_save[0][self.forcenum] = self.beta_tot
            data_to_save[1][self.forcenum] = self.beta_std_tot
            data_to_save[2][self.forcenum] = self.C_beta_tot
            data_to_save[3][self.forcenum] = self.C_beta_std_tot
            data_to_save[4][self.forcenum] = self.r_squared_visco


        elif self.fit_type == "poro":
            row_names = [["G [kPa]:"], ["ΔG [kPa]:"], ["ν:"], ["Δν:"], ["Dp [(μm)^2/s]:"], ["ΔDp [(μm)^2/s]:"],
                         ["R^2:"]]
            data_to_save = [[0 for i in range(self.forcenum + 1)] for j in range(7)]
            for i in range(self.forcenum):
                data_to_save[0][i] = self.G[i] / 1000
                data_to_save[1][i] = self.G_std[i] / 1000
                data_to_save[2][i] = self.nu[i]
                data_to_save[3][i] = self.nu_std[i]
                data_to_save[4][i] = self.D_p[i] * 10 ** 12
                data_to_save[5][i] = self.D_p_std[i] * 10 ** 12
                data_to_save[6][i] = self.r_squared_poro_values[i]
            data_to_save[0][self.forcenum] = self.G_tot / 1000
            data_to_save[1][self.forcenum] = self.G_std_tot / 1000
            data_to_save[2][self.forcenum] = self.nu_tot
            data_to_save[3][self.forcenum] = self.nu_std_tot
            data_to_save[4][self.forcenum] = self.D_p_tot * 10 ** 12
            data_to_save[5][self.forcenum] = self.D_p_std_tot * 10 ** 12
            data_to_save[6][self.forcenum] = self.r_squared_poro
        else:
            return
        data_to_save = [row1 + row2 for row1, row2 in zip(row_names, data_to_save)]

        save_ndarray_to_excel(data_to_save, save_xlsx_folder, filename, forces_headers, "Model Parameters")

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("xlsx saved successfully!")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()  # Show the popup

    def start_over_button_clicked(self):
        reply = QMessageBox.question(self, "Restart Confirmation",
                                     "Are you sure you want start over?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            """self.curr_force_sep_data = None
            self.d0 = None
            self.all_d0_vectors = []
            self.all_a_vectors = []
            self.d0_default = None
            self.all_ipts = None
            self.def_all_ipts = None
            self.all_ipt_vectors = []
            self.all_d0_vectors_merged = []
            self.all_a_vectors_merged = []
            self.all_ipt_vectors_merged = []
            self.forcenum = 0
            self.curr_force_i = 0
            self.curr_sample_i = 0
            self.duplicatenum = 0
            self.R_micron = 0
            self.n_clusters = 0
            self.final_trace_number = 0
            self.current_model_index = 0
            self.current_model = ""
            self.traceNames = []
            self.graphtitle_arr = []
            self.forcevalues_arr = []
            self.all_force_sep_AI_indices = []
            self.all_force_time_AI_indices = []
            self.all_force_sep_matrices_before_filter = []
            self.all_force_time_matrices_before_filter = []
            self.all_index_vectors_after_filtering = []
            self.all_force_sep_matrices_after_filter = []
            self.all_force_time_matrices_after_filter = []
            self.all_index_vectors_after_filtering_merged = []
            self.all_force_sep_matrices_after_filter_merged = []
            self.all_force_time_matrices_after_filter_merged = []
            self.current_force_sep_matrix = []
            self.current_force_time_matrix = []
            self.norm_x_vectors_avg = []
            self.visco_x_vectors_avg = []
            self.poro_x_vectors_avg = []
            self.norm_y_vectors_avg = []
            self.visco_y_vectors_avg = []
            self.poro_y_vectors_avg = []
            self.norm_x_vectors_std = []
            self.visco_x_vectors_std = []
            self.poro_x_vectors_std = []
            self.norm_y_vectors_std = []
            self.visco_y_vectors_std = []
            self.poro_y_vectors_std = []
            self.norm_x_vectors_UB = []
            self.visco_x_vectors_UB = []
            self.poro_x_vectors_UB = []
            self.norm_y_vectors_UB = []
            self.visco_y_vectors_UB = []
            self.poro_y_vectors_UB = []
            self.norm_x_vectors_LB = []
            self.visco_x_vectors_LB = []
            self.poro_x_vectors_LB = []
            self.norm_y_vectors_LB = []
            self.visco_y_vectors_LB = []
            self.poro_y_vectors_LB = []
            self.norm_all_matrices = []
            self.visco_all_matrices = []
            self.poro_all_matrices = []
            self.force_sep_uploaded_flag = False
            self.force_time_uploaded_flag = False
            self.current_force_signal = None
            self.current_sep_columns = None
            self.current_force_columns = None
            self.debug = False
            self.current_list_of_list_of_clusters = []
            self.current_index_vector = []
            self.F_power = 1
            self.R_power = 1
            self.F_power_name = ""
            self.R_power_name = ""
            self.d0_averages_divided = []
            self.d0_averages_final = []
            self.d0_stds_final = []
            self.current_force_sep_CSV_length = 0
            self.current_force_time_CSV_length = 0
            self.debug_upload_csv_i = 0
            self.variance_visco_vector = []
            self.variance_poro_vector = []
            self.variance_poro_vectors = []
            self.experiment_name = ""
            self.conical_indenter = False
            self.spherical_indenter = False
            self.half_angle_indenter_rad = 0
            self.tan_alpha_indenter = 0
            self.current_tab = 0
            self.trace_names_after_filtering_merged_sep = []
            self.trace_names_after_filtering_merged_time = []
            self.D_p = []
            self.G = []
            self.nu = []
            self.beta = []
            self.C_beta = []
            self.D_p_std = []
            self.G_std = []
            self.nu_std = []
            self.beta_std = []
            self.C_beta_std = []
            self.D_p_tot = 0
            self.G_tot = 0
            self.nu_tot = 0
            self.beta_tot = 0
            self.C_beta_tot = 0
            self.D_p_std_tot = 0
            self.G_std_tot = 0
            self.nu_std_tot = 0
            self.beta_std_tot = 0
            self.C_beta_std_tot = 0
            self.r_squared_visco = 0
            self.r_squared_poro = 0
            self.fit_type = ""
            self.visco_y_vectors_avg_fits = []
            self.poro_y_vectors_avg_fits = []
            self.visco_y_vectors_UB_fits = []
            self.poro_y_vectors_UB_fits = []
            self.visco_y_vectors_LB_fits = []
            self.poro_y_vectors_LB_fits = []
            self.all_G_values = []
            self.all_nu_values = []
            self.poro_fits_performed_already = False
            self.visco_fits_performed_already = False
            self.r_squared_visco_values = []
            self.r_squared_poro_values = []
            uic.loadUi('form.ui', self)
            self.FitTab.setCurrentIndex(0)
            self.FitTab.setTabEnabled(1,False)
            self.FitTab.setTabEnabled(2,False)
            self.FitTab.setTabEnabled(3,False)
            self.FitTab.setTabEnabled(4,False)"""
            self.restart = True
            if self.dark_mode_enabled:
                self.light_dark_mode_button_clicked()
            self.close()  # Close current window
            self.__init__()  # Reinitialize
            handle_design_upon_startup(self)
            self.show()  # Show the new window

        else:
            return



    def calc_contact_button_clicked(self):
        current_E_star_vector = self.all_E_star_vectors_merged[self.curr_force_i]
        if (not isinstance(current_E_star_vector, int)):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Confirmation")
            msg.setText("This instance is already calculated, do you want to override it?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            font = QFont()
            font.setPointSize(14)  # adjust as needed
            msg.setFont(font)

            # Show the message box and get the response
            response = msg.exec_()

            if response == QMessageBox.No:
                return

        self.calculate_E_star()
        self.force_curve_list_2.setEnabled(True)
        self.force_curve_list_2.setCurrentRow(0)
        self.calc_contact_button.setEnabled(False)
        self.curr_force_i = self.force_combobox_4.currentIndex()
        self.force_combobox_4.setEnabled(False)
        self.contact_slider.setEnabled(True)
        self.Save_d0_button.setEnabled(True)
        # self.model_find_d0_combobox.setEnabled(True)
        self.Indentation_depth_label.setEnabled(True)
        self.Change_model_label.setEnabled(True)
        self.show_histogram_button.setEnabled(True)
        self.Lowest_button.setEnabled(True)
        self.Highest_button.setEnabled(True)
        self.force_curve_list.setCurrentRow(-1)
        self.show_histogram_button_clicked()

    def calculate_E_star(self):
        return

    def contact_model_combobox_changed(self):
        if self.contact_model_combobox.currentIndex() == 0:
            self.adhesion_force_label.setEnabled(False)
            self.adhesion_force_combobox.setEnabled(False)
            self.adhesion_force_spinbox.setEnabled(False)
        else:
            self.adhesion_force_label.setEnabled(True)
            self.adhesion_force_combobox.setEnabled(True)
            self.adhesion_force_spinbox.setEnabled(True)


    def light_dark_mode_button_clicked(self):
        """Toggle between light and dark Fusion themes."""
        self.dark_mode_enabled = not self.dark_mode_enabled

        app = QApplication.instance()
        #app.setStyle("Fusion")

        if self.dark_mode_enabled:
            # Dark theme palette
            #app.setStyle(QStyleFactory.create('Fusion'))  # ✅ Enforce Fusion style
            dark_palette = QPalette()
            dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.WindowText, Qt.white)
            dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
            dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
            dark_palette.setColor(QPalette.ToolTipText, Qt.white)
            dark_palette.setColor(QPalette.Text, Qt.white)
            dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
            dark_palette.setColor(QPalette.ButtonText, Qt.white)
            dark_palette.setColor(QPalette.BrightText, Qt.red)
            dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
            dark_palette.setColor(QPalette.HighlightedText, Qt.black)
            app.setPalette(dark_palette)
            self.light_dark_mode_button.setText("Light\nMode")
            handle_design_upon_startup(self)
            apply_dark_mode_to_canvas(self.AI_Viewing_Graph)
            apply_dark_mode_to_canvas(self.Force_Sep_Graph)
            apply_dark_mode_to_canvas(self.Analysis_graph)
            apply_dark_mode_to_canvas(self.fit_graph)

        else:
            # Light theme palette
            """light_palette = QPalette()
            light_palette.setColor(QPalette.Window, Qt.white)
            light_palette.setColor(QPalette.WindowText, Qt.black)
            light_palette.setColor(QPalette.Base, Qt.white)
            light_palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
            light_palette.setColor(QPalette.ToolTipBase, Qt.black)
            light_palette.setColor(QPalette.ToolTipText, Qt.white)
            light_palette.setColor(QPalette.Text, Qt.black)
            light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
            light_palette.setColor(QPalette.ButtonText, Qt.black)
            light_palette.setColor(QPalette.BrightText, Qt.red)
            light_palette.setColor(QPalette.Link, QColor(0, 0, 255))
            light_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            light_palette.setColor(QPalette.HighlightedText, Qt.white)"""
            app.setStyle(QStyleFactory.create('Fusion'))  # ✅ Enforce Fusion style
            light_palette = app.style().standardPalette()
            app.setPalette(light_palette)
            apply_light_mode_to_canvas(self.AI_Viewing_Graph)
            apply_light_mode_to_canvas(self.Force_Sep_Graph)
            apply_light_mode_to_canvas(self.Analysis_graph)
            apply_light_mode_to_canvas(self.fit_graph)
            self.light_dark_mode_button.setText("Dark\nMode")
            handle_design_upon_startup(self)


    def save_clusters_csv_button_clicked(self):
        """Save clusters csv button."""
        save_csv_folder = QFileDialog.getExistingDirectory(self, "Select folder to save experiment parameters")
        if not save_csv_folder:
            return
        current_text = str(self.graphTraceAIComboBox.currentText().strip())
        if current_text.strip():  # Check if the text is not empty
            current_trace_index = int(current_text.strip()) - 1
        else:
            return
        current_cluster_index = self.clusterComboBox.currentIndex()
        if (self.force_sep_UL_radio_button.isChecked()):
            current_indices = self.all_force_sep_AI_indices[self.curr_force_i][self.curr_sample_i]
        elif (self.force_time_UL_radio_button.isChecked()):
            current_indices = self.all_force_time_AI_indices[self.curr_force_i][self.curr_sample_i]
        else:
            return
        true_indices = get_true_indices_vector(current_indices, current_cluster_index)
        #csv_file_full_path = save_csv_folder + "/" + filename
        columns_to_save = [col for i in true_indices for col in (2*i, 2*i+1)]
        matrix_to_save = self.all_force_sep_matrices_before_filter[self.curr_force_i][self.curr_sample_i][:,columns_to_save]
        headers=[f"sample {self.curr_sample_i+1} trace #{x+1} {text}" for x in true_indices for text in ("Separation [m]", "Force [N]") ]

        #header_list = self.all_headers_vectors_before_filter[self.curr_force_i][self.curr_sample_i]
        #headers = [val for i in true_indices for val in (header_list[2*i], header_list[2*i + 1])]
        sheetname = "Force vs. Sep, " + str(round(self.forcevalues_arr[self.curr_force_i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name + " sample #"+str(self.curr_sample_i)+" Cluster #"+str(current_cluster_index+1)
        filename = self.experiment_name + " " + sheetname + ".csv"
        save_ndarray_to_csv(matrix_to_save, save_csv_folder, filename, headers,
                                sheetname)
        matrix_to_save = self.all_force_time_matrices_before_filter[self.curr_force_i][self.curr_sample_i][:,columns_to_save]
        headers=[f"sample {self.curr_sample_i+1} trace #{x+1} {text}" for x in true_indices for text in ("Time [s]", "Force [N]") ]
        sheetname = "Force vs. Time, " + str(
                round(self.forcevalues_arr[self.curr_force_i] * 10 ** (-self.F_power), 2)) + " " + self.F_power_name + " sample #"+str(self.curr_sample_i+1)+" Cluster #"+str(current_cluster_index+1)
        filename = self.experiment_name + " " + sheetname + ".csv"
        save_ndarray_to_csv(matrix_to_save, save_csv_folder, filename, headers,
                                sheetname)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)  # Options: Information, Warning, Critical, Question
        msg_box.setWindowTitle("Confirmation")
        msg_box.setText("cluster csv saved successfully!")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()  # Show the popup






if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    light_palette = app.style().standardPalette()
    app.setPalette(light_palette)
    window = FindD0App()
    handle_design_upon_startup(window)

    window.show()
    sys.exit(app.exec_())