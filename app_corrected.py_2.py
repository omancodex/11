"""
CHPE4412 Process Heat Transfer Project
Interactive Dashboard for Shell-and-Tube Heat Exchanger Design
Methods: Kern + simplified Bell-Delaware correction framework
Case study: cooling hot petroleum fraction using cooling water
Important academic note:
This dashboard is for educational engineering design. Final industrial designs must be checked
against TEMA/ASME standards, mechanical design, vibration, fouling history, and plant constraints.
"""
import math
import os
import base64
from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="Shell-and-Tube HX Dashboard", layout="wide")


# Apply the same background image to every screen/page
def get_base64_of_bin_file(filename):
    import sys
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    bin_file = os.path.join(base_path, filename)
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

img_base64 = get_base64_of_bin_file("background.jpg")

if img_base64:
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.75);
        border-radius: 15px;
        padding-top: 2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    </style>
    """, unsafe_allow_html=True)


# Initialize session state for screen navigation
_screen = st.session_state.get('screen', 'welcome')
if 'screen' not in st.session_state:
    st.session_state.screen = 'welcome'
# Defaults dictionary
defaults = {
    'mh': 20.0, 'Th_in': 150.0, 'Th_out': 90.0, 'cph': 2200.0, 'rhoh': 800.0, 'muh': 0.003, 'kh': 0.13,
    'Tc_in': 25.0, 'Tc_out': 40.0, 'cpc': 4180.0, 'rhoc': 997.0, 'muc': 0.00089, 'kc': 0.60,
    'Ds': 0.6, 'do_mm': 19.05, 'di_mm': 16.0, 'L': 4.88, 'pitch_ratio': 1.25,
    'layout': 'Triangular 30°/60°', 'passes': 2, 'tube_count_mode': 'Auto from required area',
    'Nt_manual': 124, 'baffle_cut': 25, 'baffle_spacing_ratio': 0.40, 'kwall': 45.0,
    'Rfi': 0.0002, 'Rfo': 0.0005, 'leakage_quality': 'Normal clearances', 'bypass_quality': 'Normal bypass',
    'sealing_strips': 2, 'design_margin': 15, 'Nt_factor': 1.0, 'dp_allow_tube': 100.0, 'dp_allow_shell': 100.0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
# -----------------------------
# Helper functions
# -----------------------------
def safe_div(a, b, default=np.nan):
    try:
        if abs(b) < 1e-30:
            return default
        return a / b
    except Exception:
        return default
def lmtd_counter_current(Th_in, Th_out, Tc_in, Tc_out):
    dT1 = Th_in - Tc_out
    dT2 = Th_out - Tc_in
    if dT1 <= 0 or dT2 <= 0:
        return np.nan, dT1, dT2
    if abs(dT1 - dT2) < 1e-9:
        return dT1, dT1, dT2
    return (dT1 - dT2) / math.log(dT1 / dT2), dT1, dT2
def ft_1_shell_2n_tube(Th_in, Th_out, Tc_in, Tc_out):
    # Standard approximate F correction for 1 shell pass and even tube passes.
    R = safe_div(Th_in - Th_out, Tc_out - Tc_in)
    P = safe_div(Tc_out - Tc_in, Th_in - Tc_in)
    if not np.isfinite(R) or not np.isfinite(P) or P <= 0 or P >= 1:
        return np.nan, R, P
    try:
        S = math.sqrt(R**2 + 1)
        if abs(R - 1) < 1e-8:
            # Limit expression when R approximately equals 1
            numerator = S * P
            denominator = 2 - P * (R + 1 - S)
            return np.nan if denominator <= 0 else 0.95, R, P
        num = S / (R - 1) * math.log((1 - P) / (1 - R * P))
        den_arg = (2 - P * (R + 1 - S)) / (2 - P * (R + 1 + S))
        if den_arg <= 0:
            return np.nan, R, P
        den = math.log(den_arg)
        F = num / den
        if F < 0 or F > 1.2:
            return np.nan, R, P
        return F, R, P
    except Exception:
        return np.nan, R, P
def dittus_boelter(Re, Pr, k, Di, heating=True):
    if Re < 10000:
        # Simple educational fallback: Sieder-Tate laminar/transition not implemented fully.
        Nu = 3.66 if Re < 2300 else 0.023 * Re**0.8 * Pr**0.4
    else:
        n = 0.4 if heating else 0.3
        Nu = 0.023 * Re**0.8 * Pr**n
    h = safe_div(Nu * k, Di)
    return Nu, h
def friction_factor_smooth(Re):
    if Re <= 0 or not np.isfinite(Re):
        return np.nan
    if Re < 2300:
        return 64 / Re
    return 0.3164 / Re**0.25
def equivalent_diameter(layout, pitch, do):
    if layout == "Triangular 30°/60°":
        return 1.10 * (pitch**2 - 0.917 * do**2) / do
    # square pitch
    return 1.27 * (pitch**2 - 0.785 * do**2) / do
def kern_shell_h(m_shell, cp, mu, k, do, shell_d, pitch, baffle_spacing, layout):
    de = equivalent_diameter(layout, pitch, do)
    # Cross-flow area approximation, Kern style
    As = max(shell_d * baffle_spacing * (pitch - do) / pitch, 1e-9)
    Gs = m_shell / As
    Re_s = Gs * de / mu
    Pr_s = cp * mu / k
    # Kern jH-type correlation: Nu = 0.36 Re^0.55 Pr^(1/3)
    Nu_s = 0.36 * max(Re_s, 1e-9)**0.55 * max(Pr_s, 1e-9)**(1/3)
    h_s = Nu_s * k / de
    return de, As, Gs, Re_s, Pr_s, Nu_s, h_s
def delaware_pressure_drop_reference(
    Ds, Db, B, baffle_cut, pitch, do, Nt, Gs, rho, Re_s,
    layout, Nb, Rl, Rb, Rs
):
    """
    Slide-based Bell-Delaware shell-side pressure drop:
    ΔPf = [(nb-1)ΔPideal RB + nb ΔPw,ideal]RL
          + 2ΔPideal(1 + Ncw/Nc)RBRS
    """
    Bc = baffle_cut / 100.0
    # projected tube pitch P'T
    if layout == "Square 90°":
        Pt_eff = pitch
    elif layout == "Triangular 30°/60°":
        Pt_eff = pitch * math.cos(math.radians(30))
    else:  # rotated square
        Pt_eff = pitch * math.cos(math.radians(45))
    Pt_eff = max(Pt_eff, 1e-9)
    # number of tube rows in cross-flow
    Nc = max(Ds * (1 - 2 * Bc) / Pt_eff, 1e-9)
    # ideal friction factor
    f_ideal = friction_factor_smooth(Re_s)
    # ideal cross-flow pressure drop per central baffle space, Pa
    dp_ideal = 2 * f_ideal * Nc * Gs**2 / max(rho, 1e-9)
    # baffle window angle
    theta_ds = 2 * math.acos(max(-1, min(1, 1 - 2 * Bc)))
    # fraction of tubes in one window
    Fc = max(0, min(1, 1 - 2 * Bc))
    Fw = max(0, (1 - Fc) / 2)
    # window flow area Sw, m2
    Sw = (1/8) * Ds**2 * (theta_ds - math.sin(theta_ds)) \
         - (1/4) * Nt * Fw * math.pi * do**2
    Sw = max(Sw, 1e-9)
    # mass flow through shell side
    m_shell = Gs * max(B * (Ds - Db), 1e-9)
    # effective number of tube rows in one baffle window
    Ncw = max(0.8 * Bc * Ds / Pt_eff, 1e-9)
    # window ideal pressure drop, Pa
    dp_w_ideal = ((2 + 0.6 * Ncw) * m_shell**2) / (2 * rho * Sw**2)
    # final slide equations, Pa
    dp_c = (Nb - 1) * dp_ideal * Rb * Rl
    dp_w = Nb * dp_w_ideal * Rl
    dp_e = 2 * dp_ideal * (1 + Ncw / Nc) * Rb * Rs
    dp_f = dp_c + dp_w + dp_e
    return {
        "Nc": Nc,
        "Ncw": Ncw,
        "Sw": Sw,
        "dp_ideal": dp_ideal / 1000,
        "dp_w_ideal": dp_w_ideal / 1000,
        "dp_c": dp_c / 1000,
        "dp_w": dp_w / 1000,
        "dp_e": dp_e / 1000,
        "dp_f": dp_f / 1000
    }
def overall_U(hi, ho, di, do, kwall, Rfi, Rfo):
    # Overall coefficient based on outside area
    # 1/Uo = 1/ho + Rfo + do*ln(do/di)/(2*k_wall) + (do/di)*(Rfi + 1/hi)
    wall = do * math.log(do / di) / (2 * kwall)
    invU = safe_div(1, ho) + Rfo + wall + (do / di) * (Rfi + safe_div(1, hi))
    return safe_div(1, invU)
def tube_count_from_area(A, do, L):
    return math.ceil(max(A, 0) / (math.pi * do * L))
def shell_diameter_estimate(Nt, pitch, passes, layout):
    # Educational approximate bundle diameter and shell diameter estimate.
    # Conservative allowance factor increases with tube passes.
    if Nt <= 0:
        return np.nan, np.nan
    layout_factor = 0.87 if layout == "Triangular 30°/60°" else 0.79
    pass_factor = {1: 1.00, 2: 1.08, 4: 1.18, 6: 1.25, 8: 1.32}.get(passes, 1.15)
    Db = pitch * math.sqrt(Nt / layout_factor) * pass_factor
    Ds = Db + 0.075  # clearance allowance, m
    return Db, Ds
def tube_side_calcs(m, rho, mu, cp, k, di, do, L, Nt, passes):
    tubes_per_pass = max(Nt / passes, 1)
    area_one_tube = math.pi * di**2 / 4
    flow_area = tubes_per_pass * area_one_tube
    v = safe_div(m, rho * flow_area)
    Re = rho * v * di / mu
    Pr = cp * mu / k
    Nu, h = dittus_boelter(Re, Pr, k, di, heating=True)
    f = friction_factor_smooth(Re)
    # Pressure drop: tube friction + approximate return losses
    dp_fric = f * (L * passes / di) * rho * v**2 / 2
    dp_return = 4 * passes * rho * v**2 / 2
    return flow_area, v, Re, Pr, Nu, h, f, dp_fric + dp_return
def shell_dp_kern(Gs, rho, de, shell_d, baffle_spacing, f_s=0.2):
    # Educational approximation for shell-side pressure drop.
    Nb = max(int(shell_d / max(baffle_spacing, 1e-6)), 1)
    dp = f_s * Gs**2 * shell_d * (Nb + 1) / (2 * rho * de)
    return Nb, dp
def delaware_corrections(baffle_cut, sealing_strips, leakage_quality, bypass_quality, Re_s):
    # Simplified correction framework for dashboard learning.
    # Values are bounded and represent qualitative Bell-Delaware effects.
    Jc = 1.0 - 0.004 * abs(baffle_cut - 25)          # baffle window/crossflow correction
    Jl = {"Tight clearances": 0.92, "Normal clearances": 0.82, "Large leakage": 0.70}[leakage_quality]
    Jb_base = {"Low bypass": 0.92, "Normal bypass": 0.82, "High bypass": 0.68}[bypass_quality]
    Jb = min(0.98, Jb_base + 0.02 * sealing_strips)
    Jr = 1.0 if Re_s >= 100 else max(0.65, 0.75 + 0.0025 * Re_s)
    Js = 0.95  # unequal end spacing correction default
    Jtotal = np.prod([Jc, Jl, Jb, Jr, Js])
    return {"Jc": max(0.55, min(Jc, 1.0)), "Jl": Jl, "Jb": Jb, "Jr": Jr, "Js": Js, "Jtotal": Jtotal}
def projected_tube_pitch(layout, pitch):
    """Projected tube pitch P'_T from the lecture slide."""
    if "Square" in layout:
        return pitch
    return pitch * math.cos(math.radians(30))
def estimate_clearance(Ds):
    # Ds in meters → convert to mm
    Ds_mm = Ds * 1000
    if Ds_mm < 1000:
        return 0.02  # 20 mm
    elif Ds_mm < 2000:
        return 0.03  # 30 mm
    else:
        return 0.04  # 40 mm
def delaware_flow_areas(Ds, B, do, pitch, Nt, baffle_cut, layout, sealing_strips):
    """Bell-Delaware flow areas and ratios from the lecture-slide equations."""
    Bc = baffle_cut / 100.0
    Ptp = projected_tube_pitch(layout, pitch)
    clearance = estimate_clearance(Ds)
    Dotl = Ds - clearance
    delta_tb = 0.0004 if do > 0.03175 else 0.0002
    delta_sb = (0.8 + 0.002 * (Ds * 1000.0)) / 1000.0
    theta_ds = 2.0 * math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * Bc)))
    Fc = 1.0 - (theta_ds - math.sin(theta_ds)) / (2.0 * math.pi)
    Fw = (1.0 - Fc) / 2.0
    Sm = B * (
    (Ds - Dotl)
    + ((Dotl - do) / max(Ptp, 1e-12)) * (pitch - do)
    )
    Stb = 0.5 * math.pi * do * delta_tb * Nt * (1.0 + Fc)
    Ssb = Ds * delta_sb * (math.pi - 0.5 * theta_ds)
    Sb = B * max(Ds - Dotl, 0.0)
    Sw = 0.125 * Ds**2 * (theta_ds - math.sin(theta_ds)) - 0.25 * Nt * Fw * math.pi * do**2
    Sm = max(Sm, 1e-12)
    Sw = max(Sw, 1e-12)
    rs = safe_div(Ssb, Ssb + Stb, 0.0)
    rl = safe_div(Ssb + Stb, Sm, 0.0)
    Nc = max(1.0, Ds * (1.0 - 2.0 * Bc) / max(Ptp, 1e-12))
    Ncw = max(1.0, 0.8 * Bc * Ds / max(Ptp, 1e-12))
    Nss = sealing_strips
    rss = safe_div(Nss, Nc, 0.0)
    return {
        "Ptp": Ptp, "Dotl": Dotl, "delta_tb": delta_tb, "delta_sb": delta_sb,
        "theta_ds": theta_ds, "Fc": Fc, "Fw": Fw,
        "Sm": Sm, "Stb": Stb, "Ssb": Ssb, "Sb": Sb, "Sw": Sw,
        "rs": rs, "rl": rl, "Nc": Nc, "Ncw": Ncw, "Nss": Nss, "rss": rss,
    }
def delaware_heat_transfer_corrections(areas, Re_s, Nb):
    """Heat-transfer correction factors Jc, Jl, Jb, Js, Jr from the lecture slides."""
    Fc, rs, rl, rss = areas["Fc"], areas["rs"], areas["rl"], areas["rss"]
    Sm, Sb = areas["Sm"], areas["Sb"]
    Nc, Ncw = areas["Nc"], areas["Ncw"]
    Jc = 0.55 + 0.72 * Fc
    Jl = 0.44 * (1.0 - rs) + (1.0 - 0.44 * (1.0 - rs)) * math.exp(-2.2 * rl)
    Cj = 1.25 if Re_s >= 100 else 1.35
    Jb = 1.0 if rss >= 0.5 else math.exp(-Cj * safe_div(Sb, Sm, 0.0) * (1.0 - (2.0 * rss) ** (1.0 / 3.0)))
    Js = 1.0  # Bin = Bout = B in this dashboard; slide then gives Js = 1
    Nct = max((Nb + 1.0) * (Nc + Ncw), 1.0)
    Jr_low = (10.0 / Nct) ** 0.18
    if Re_s >= 100:
        Jr = 1.0
    elif Re_s <= 20:
        Jr = Jr_low
    else:
        Jr = Jr_low + (1.0 - Jr_low) * (Re_s - 20.0) / 80.0
    Jc = max(0.0, min(Jc, 1.0)); Jl = max(0.0, min(Jl, 1.0))
    Jb = max(0.0, min(Jb, 1.0)); Jr = max(0.0, min(Jr, 1.0))
    Jtotal = Jc * Jl * Jb * Js * Jr
    return {"Jc": Jc, "Jl": Jl, "Jb": Jb, "Js": Js, "Jr": Jr, "Jtotal": Jtotal, "Nct": Nct}
def delaware_pressure_drop(m_shell, rho, Gs_fallback, Re_s, Ds, B, do, pitch, Nt, baffle_cut, layout, sealing_strips, Nb):
    """Slide-based Delaware shell-side pressure-drop split. Returned ΔP values are Pa."""
    areas = delaware_flow_areas(Ds, B, do, pitch, Nt, baffle_cut, layout, sealing_strips)
    Sm, Sw, Sb = areas["Sm"], areas["Sw"], areas["Sb"]
    rs, rl, rss = areas["rs"], areas["rl"], areas["rss"]
    Nc, Ncw = areas["Nc"], areas["Ncw"]
    Nb = max(float(Nb), 1.0)
    Gm = m_shell / Sm if Sm > 0 else Gs_fallback
    # The lecture slide requires f_ideal from a tube-bank chart. The dashboard estimates it
    # with the existing friction factor correlation because the chart is not coded.
    f_ideal = friction_factor_smooth(max(Re_s, 1e-9))
    if not np.isfinite(f_ideal):
        f_ideal = 0.2
    dp_ideal = 2.0 * f_ideal * Nc * Gm**2 / max(rho, 1e-12)
    dp_w_ideal = ((2.0 + 0.6 * Ncw) * m_shell**2) / max(2.0 * rho * Sm * Sw, 1e-12)
    p_exp = 0.8 - 0.15 * (1.0 + rs)
    Rl = math.exp(-1.33 * (1.0 + rs) * (max(rl, 0.0) ** p_exp))
    Cr = 3.7 if Re_s >= 100 else 4.5
    Rb = 1.0 if rss >= 0.5 else math.exp(-Cr * safe_div(Sb, Sm, 0.0) * (1.0 - (2.0 * rss) ** (1.0 / 3.0)))
    Rs = 1.0  # Bin = Bout = B in this dashboard; slide then gives Rs = 1
    Rl = max(0.0, min(Rl, 1.0)); Rb = max(0.0, min(Rb, 1.0))
    dp_cross = (Nb - 1.0) * dp_ideal * Rl * Rb
    dp_window = Nb * dp_w_ideal * Rl
    dp_end = 2.0 * dp_ideal * (1.0 + safe_div(Ncw, Nc, 0.0)) * Rb * Rs
    dp_total = dp_cross + dp_window + dp_end
    return {
        **areas, "Rl": Rl, "Rb": Rb, "Rs": Rs, "f_ideal": f_ideal, "Gm": Gm,
        "dp_ideal": dp_ideal, "dp_w_ideal": dp_w_ideal,
        "dp_cross": dp_cross, "dp_window": dp_window, "dp_end": dp_end, "dp_total": dp_total,
    }
def status_badge(value, low=None, high=None, reverse=False):
    if not np.isfinite(value):
        return "⚠️ Not valid"
    ok = True
    if low is not None and value < low:
        ok = False
    if high is not None and value > high:
        ok = False
    if reverse:
        ok = not ok
    return "✅ Acceptable" if ok else "⚠️ Check design"
# -----------------------------
# Screen Navigation Logic
# -----------------------------
if st.session_state.screen == 'welcome':
    # Welcome Screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; max-width: 800px; margin: 0 auto;">
        <h1>Welcome to Heat Exchanger Design Assistant</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem; line-height: 1.6;">
            This interactive tool helps you analyze and design shell-and-tube heat exchangers using both Kern and Delaware methods. Perfect for engineering students and professionals working on heat transfer projects.
        </p>
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; padding: 2rem; margin: 2rem 0; border-left: 5px solid #1f77b4; text-align: left; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3>How to Use This Tool</h3>
            <ol style="list-style: none; padding: 0;">
                <li style="margin: 0.5rem 0; padding: 0.5rem 0; border-bottom: 1px solid #dee2e6; position: relative; padding-left: 2rem;">Understand the purpose of the tool</li>
                <li style="margin: 0.5rem 0; padding: 0.5rem 0; border-bottom: 1px solid #dee2e6; position: relative; padding-left: 2rem;">Click Start Calculator to open the input panel</li>
                <li style="margin: 0.5rem 0; padding: 0.5rem 0; border-bottom: 1px solid #dee2e6; position: relative; padding-left: 2rem;">Enter operating conditions, fluid properties, and geometry</li>
                <li style="margin: 0.5rem 0; padding: 0.5rem 0; border-bottom: 1px solid #dee2e6; position: relative; padding-left: 2rem;">Click Analyze Design to run calculations</li>
                <li style="margin: 0.5rem 0; padding: 0.5rem 0; position: relative; padding-left: 2rem;">Review results, charts, and recommendations</li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Calculator", type="secondary", key="start_btn"):
            st.session_state.screen = 'input'
            st.rerun()
    st.stop()

elif st.session_state.screen == 'input':
    # Input Screen
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("← Back to Welcome", key="back_to_welcome_btn"):
            st.session_state.screen = 'welcome'
            st.rerun()
            
    st.title("Heat Exchanger Design Inputs")
    st.markdown("Enter the operating conditions, fluid properties, and geometry settings below.")

    # Section 1 — Operating Conditions
    st.subheader("Operating Conditions")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.Th_in = st.number_input("Hot inlet temperature, °C", 30.0, 400.0, st.session_state.Th_in, step=5.0)
        st.session_state.Th_out = st.number_input("Hot outlet temperature, °C", 20.0, 350.0, st.session_state.Th_out, step=5.0)
        st.session_state.mh = st.number_input("Hot fluid mass flow rate, kg/s", 1.0, 500.0, st.session_state.mh, step=1.0)
    with col2:
        st.session_state.Tc_in = st.number_input("Cold inlet temperature, °C", 0.0, 100.0, st.session_state.Tc_in, step=1.0)
        st.session_state.Tc_out = st.number_input("Cold outlet temperature, °C", 1.0, 120.0, st.session_state.Tc_out, step=1.0)
        st.session_state.cpc = st.number_input("Cold Cp, J/kg.K", 1000.0, 6000.0, st.session_state.cpc, step=10.0)

    # Section 2 — Fluid Properties
    st.subheader("Fluid Properties")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Shell Side (Hot Fluid)**")
        st.session_state.cph = st.number_input("Hot Cp, J/kg.K", 500.0, 6000.0, st.session_state.cph, step=100.0)
        st.session_state.rhoh = st.number_input("Hot density, kg/m³", 300.0, 1500.0, st.session_state.rhoh, step=10.0)
        st.session_state.muh = st.number_input("Hot viscosity, Pa.s", 0.0001, 0.1, st.session_state.muh, step=0.0001, format="%.5f")
        st.session_state.kh = st.number_input("Hot thermal conductivity, W/m.K", 0.02, 1.0, st.session_state.kh, step=0.01)
    with col2:
        st.markdown("**Tube Side (Cold Fluid)**")
        st.session_state.rhoc = st.number_input("Cold density, kg/m³", 500.0, 1200.0, st.session_state.rhoc, step=1.0)
        st.session_state.muc = st.number_input("Cold viscosity, Pa.s", 0.0001, 0.01, st.session_state.muc, step=0.00001, format="%.5f")
        st.session_state.kc = st.number_input("Cold thermal conductivity, W/m.K", 0.05, 1.0, st.session_state.kc, step=0.01)

    # Section 3 — Geometry Settings
    st.subheader("Geometry Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.Ds = st.number_input("Shell inside diameter, m", 0.2, 3.0, st.session_state.Ds, step=0.05)
        st.session_state.do_mm = st.number_input("Tube outside diameter, mm", 10.0, 50.0, st.session_state.do_mm, step=0.5)
        st.session_state.L = st.number_input("Tube length, m", 1.0, 12.0, st.session_state.L, step=0.1)
        st.session_state.pitch_ratio = st.number_input("Pitch ratio, Pt/Do", 1.10, 2.00, st.session_state.pitch_ratio, step=0.05)
    with col2:
        st.session_state.di_mm = st.number_input("Tube inside diameter, mm", 8.0, 48.0, st.session_state.di_mm, step=0.5)
        st.session_state.layout = st.selectbox("Tube layout", ["Triangular 30°/60°", "Square 90°"])
        st.session_state.passes = st.selectbox("Tube passes", [1, 2, 4, 6, 8], index=1)
        st.session_state.Nt_manual = st.number_input("Manual number of tubes", 20, 2000, st.session_state.Nt_manual, step=1)

    # Additional settings (can be in expander if needed, but for now inline)
    st.subheader("Additional Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.baffle_cut = st.slider("Baffle cut, %", 15, 45, st.session_state.baffle_cut)
        st.session_state.baffle_spacing_ratio = st.slider("Baffle spacing ratio, B/Ds", 0.20, 1.00, st.session_state.baffle_spacing_ratio, step=0.05)
        st.session_state.kwall = st.number_input("Tube wall thermal conductivity, W/m.K", 5.0, 400.0, st.session_state.kwall, step=5.0)
        st.session_state.Rfi = st.number_input("Tube-side fouling resistance, m².K/W", 0.0, 0.005, st.session_state.Rfi, step=0.0001, format="%.5f")
    with col2:
        st.session_state.Rfo = st.number_input("Shell-side fouling resistance, m².K/W", 0.0, 0.005, st.session_state.Rfo, step=0.0001, format="%.5f")
        st.session_state.leakage_quality = st.selectbox("Delaware leakage condition", ["Tight clearances", "Normal clearances", "Large leakage"], index=1)
        st.session_state.bypass_quality = st.selectbox("Delaware bypass condition", ["Low bypass", "Normal bypass", "High bypass"], index=1)
        st.session_state.sealing_strips = st.slider("Number of sealing strip pairs", 0, 6, st.session_state.sealing_strips)
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.design_margin = st.slider("Area design margin, %", 0, 50, st.session_state.design_margin)
        st.session_state.Nt_factor = st.slider("Tube count adjustment factor", 0.3, 1.2, st.session_state.Nt_factor)
    with col2:
        st.session_state.dp_allow_tube = st.number_input("Allowable tube-side ΔP, kPa", 5.0, 500.0, st.session_state.dp_allow_tube, step=5.0)
        st.session_state.dp_allow_shell = st.number_input("Allowable shell-side ΔP, kPa", 5.0, 500.0, st.session_state.dp_allow_shell, step=5.0)

    # Analyze Design button
    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("Analyze Design", type="primary", key="analyze_btn", use_container_width=True):
            st.session_state.screen = 'result'
            st.rerun()

    # Hidden block to assign to local variables
    pythondi_mm = st.session_state.di_mm
    layout = st.session_state.layout
    tube_count_mode = "Manual input"
    baffle_cut = st.session_state.baffle_cut
    baffle_spacing_ratio = st.session_state.baffle_spacing_ratio
    kwall = st.session_state.kwall
    Rfi = st.session_state.Rfi
    Rfo = st.session_state.Rfo
    leakage_quality = st.session_state.leakage_quality
    bypass_quality = st.session_state.bypass_quality
    sealing_strips = st.session_state.sealing_strips
    design_margin = st.session_state.design_margin
    Nt_factor = st.session_state.Nt_factor
    dp_allow_tube = st.session_state.dp_allow_tube
    dp_allow_shell = st.session_state.dp_allow_shell
    st.stop()

elif st.session_state.screen == 'result':
    # Result Screen
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("← Back to Inputs", key="back_btn"):
            st.session_state.screen = 'input'
            st.rerun()
    st.title("Shell-and-Tube Heat Exchanger Design Dashboard")
    st.subheader("Kern Method + Delaware Method | Industrial-style interactive project dashboard")
    st.info("Case study: cooling hot petroleum fraction on the shell side using cooling water on the tube side. The dashboard recalculates instantly when inputs change.")

    # Read values from session state
    mh = st.session_state.mh
    Th_in = st.session_state.Th_in
    Th_out = st.session_state.Th_out
    cph = st.session_state.cph
    rhoh = st.session_state.rhoh
    muh = st.session_state.muh
    kh = st.session_state.kh
    Tc_in = st.session_state.Tc_in
    Tc_out = st.session_state.Tc_out
    cpc = st.session_state.cpc
    rhoc = st.session_state.rhoc
    muc = st.session_state.muc
    kc = st.session_state.kc
    Ds = st.session_state.Ds
    do_mm = st.session_state.do_mm
    di_mm = st.session_state.di_mm
    L = st.session_state.L
    pitch_ratio = st.session_state.pitch_ratio
    layout = st.session_state.layout
    passes = st.session_state.passes
    tube_count_mode = "Manual input"
    Nt_manual = st.session_state.Nt_manual
    baffle_cut = st.session_state.baffle_cut
    baffle_spacing_ratio = st.session_state.baffle_spacing_ratio
    kwall = st.session_state.kwall
    Rfi = st.session_state.Rfi
    Rfo = st.session_state.Rfo
    leakage_quality = st.session_state.leakage_quality
    bypass_quality = st.session_state.bypass_quality
    sealing_strips = st.session_state.sealing_strips
    design_margin = st.session_state.design_margin
    Nt_factor = st.session_state.Nt_factor
    dp_allow_tube = st.session_state.dp_allow_tube
    dp_allow_shell = st.session_state.dp_allow_shell
    # -----------------------------
    # Main calculations
    # -----------------------------
    do = do_mm / 1000
    ndi = di_mm / 1000
    pitch = pitch_ratio * do
    Q_hot = mh * cph * (Th_in - Th_out)
    mc = safe_div(Q_hot, cpc * (Tc_out - Tc_in))
    lmtd, dT1, dT2 = lmtd_counter_current(Th_in, Th_out, Tc_in, Tc_out)
    F, R, P = ft_1_shell_2n_tube(Th_in, Th_out, Tc_in, Tc_out)
    F_used = F if np.isfinite(F) and F > 0 else 0.85
    DT_eff = lmtd * F_used if np.isfinite(lmtd) else np.nan
    # preliminary U guess for initial geometry
    U_guess = 350.0
    A_prelim = safe_div(Q_hot, U_guess * DT_eff)
    Nt_prelim_auto = tube_count_from_area(A_prelim * (1 + design_margin / 100), do, L)
    if tube_count_mode == "Manual input":
        Nt_prelim = Nt_manual
    else:
        Nt_prelim = Nt_prelim_auto
    Db_est, Ds_est = shell_diameter_estimate(Nt_prelim, pitch, passes, layout)
    # ===============================
    # ITERATION LOOP (Industrial Level)
    # ===============================
    max_iter = 20
    tolerance = 0.01
    Nt = Nt_prelim
    Ds = Ds_est
    for i in range(max_iter):
        # Recalculate geometry
        Db, Ds = shell_diameter_estimate(Nt, pitch, passes, layout)
        B = baffle_spacing_ratio * Ds
        # Tube + Shell calculations
        (tube_A, vt, Re_t, Pr_t, Nu_t, hi, ft, dp_tube) = tube_side_calcs(
        mc, rhoc, muc, cpc, kc, ndi, do, L, Nt, passes
        )
        (de, As, Gs, Re_s, Pr_s, Nu_s, ho_kern) = kern_shell_h(
            mh, cph, muh, kh, do, Ds, pitch, B, layout
        )
        Nb, dp_shell_kern = shell_dp_kern(Gs, rhoh, de, Ds, B)
        # Delaware correction
        # Delaware heat-transfer correction using slide-based Bell-Delaware factors
        Nb_iter, _ = shell_dp_kern(Gs, rhoh, de, Ds, B)
        pd_iter = delaware_pressure_drop(
            mh, rhoh, Gs, Re_s, Ds, B, do, pitch, Nt,
            baffle_cut, layout, sealing_strips, Nb_iter
        )
        corr_iter = delaware_heat_transfer_corrections(pd_iter, Re_s, Nb_iter)
        ho = ho_kern * corr_iter["Jtotal"]
        Uo = overall_U(hi, ho, ndi, do, kwall, Rfi, Rfo)
        A_required = safe_div(Q_hot, Uo * DT_eff)
        # Update tube count
        Nt_new = tube_count_from_area(A_required * (1 + design_margin / 100), do, L)
        # Convergence check
        if abs(Nt_new - Nt) / Nt < tolerance:
            break
        Nt = Nt_new
    B = baffle_spacing_ratio * Ds
    # tube and shell side using preliminary tube count and shell diameter
    (tube_A, vt, Re_t, Pr_t, Nu_t, hi, ft, dp_tube) = tube_side_calcs(mc, rhoc, muc, cpc, kc, ndi, do, L, Nt, passes)
    (de, As, Gs, Re_s, Pr_s, Nu_s, ho_kern) = kern_shell_h(mh, cph, muh, kh, do, Ds, pitch, B, layout)
    Nb, dp_shell_kern = shell_dp_kern(Gs, rhoh, de, Ds, B)
    Uo_kern = overall_U(hi, ho_kern, ndi, do, kwall, Rfi, Rfo)
    # ===============================
    # Delaware Method: Flow Areas, Correction Factors, and Pressure Drop
    # Based on the user's Delaware lecture-slide equations
    # ===============================
    pd_results = delaware_pressure_drop(
        mh, rhoh, Gs, Re_s, Ds, B, do, pitch, Nt, baffle_cut, layout, sealing_strips, Nb
    )
    Sm = pd_results["Sm"]
    Stb = pd_results["Stb"]
    Ssb = pd_results["Ssb"]
    Sb = pd_results["Sb"]
    Sw = pd_results["Sw"]
    rs = pd_results["rs"]
    rl = pd_results["rl"]
    rss = pd_results["rss"]
    Nc = pd_results["Nc"]
    Ncw = pd_results["Ncw"]
    Nss = pd_results["Nss"]
    Ptp = pd_results["Ptp"]
    theta_ds = pd_results["theta_ds"]
    Fc = pd_results["Fc"]
    Fw = pd_results["Fw"]
    Dotl = pd_results["Dotl"]
    delta_tb = pd_results["delta_tb"]
    delta_sb = pd_results["delta_sb"]
    corr = delaware_heat_transfer_corrections(pd_results, Re_s, Nb)
    Jc = corr["Jc"]
    Jl = corr["Jl"]
    Jb = corr["Jb"]
    Js = corr["Js"]
    Jr = corr["Jr"]
    J_total = corr["Jtotal"]
    Nct = corr["Nct"]
    Rl = pd_results["Rl"]
    Rb = pd_results["Rb"]
    Rs = pd_results["Rs"]
    # Use reference-based Delaware pressure drop (from slides)
    dp_ref = delaware_pressure_drop_reference(
        Ds=Ds,
        Db=Db,
        B=B,
        baffle_cut=baffle_cut,
        pitch=pitch,
        do=do,
        Nt=Nt,
        Gs=Gs,
        rho=rhoh,
        Re_s=Re_s,
        layout=layout,
        Nb=Nb,
        Rl=Rl,
        Rb=Rb,
        Rs=Rs
    )
    # Extract values (exactly like your reference equations)
    dp_ideal = dp_ref["dp_ideal"]
    dp_w_ideal = dp_ref["dp_w_ideal"]
    dp_cross_delaware = dp_ref["dp_c"]     # ΔPc
    dp_window_delaware = dp_ref["dp_w"]    # ΔPw
    dp_end_delaware = dp_ref["dp_e"]       # ΔPe
    dp_shell_delaware = dp_ref["dp_f"]     # ΔPf (final total)
    ho_delaware_full = ho_kern * J_total
    Uo_delaware_full = overall_U(hi, ho_delaware_full, ndi, do, kwall, Rfi, Rfo)
    A_delaware_full = safe_div(Q_hot, Uo_delaware_full * DT_eff)
    A_kern = safe_div(Q_hot, Uo_kern * DT_eff)
    Nt_kern = tube_count_from_area(A_kern * (1 + design_margin / 100), do, L)
    Db_k, Ds_k = shell_diameter_estimate(Nt_kern, pitch, passes, layout)
    ho_delaware = ho_delaware_full
    Uo_delaware = Uo_delaware_full
    A_delaware = A_delaware_full
    Nt_delaware = tube_count_from_area(A_delaware * (1 + design_margin / 100), do, L)
    # Delaware pressure drop follows the slide split: ΔPc + ΔPw + ΔPe.
    # -----------------------------
    # Welcome Screen
    # -----------------------------
    tabs = st.tabs(["Overview", "Thermal Design", "Kern Method", "Delaware Method", 'Iteration & Optimization', "Comparison", "Sensitivity",'Design check', "Final Report"])
    with tabs[0]:
        st.header("Project Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Heat Duty", f"{Q_hot/1e6:.3f} MW")
        c2.metric("Required Water Flow", f"{mc:.2f} kg/s")
        c3.metric("Effective ΔT", f"{DT_eff:.2f} °C")
        st.markdown("""
        **Design problem:** A hot petroleum fraction must be cooled before downstream processing. Cooling water is selected as the utility.
        **Why this case is industrial:** refinery and petrochemical plants commonly cool hydrocarbon process streams using shell-and-tube exchangers because they are robust, maintainable, and suitable for high flow rates.
        **Dashboard purpose:** convert heat-transfer equations into a digital design tool that allows calculation, comparison, visualization, sensitivity analysis, and decision-making.
        """)
        st.subheader("Assumptions")
        st.write(pd.DataFrame({
            "Assumption": [
                "Steady-state operation",
                "No heat loss to surroundings",
                "Constant average fluid properties",
                "Single-phase sensible heat transfer",
                "Counter-current correction handled using F factor",
                "Tube-side water, shell-side petroleum fraction",
                "Educational Delaware corrections are used for interactive comparison"
            ]
        }))
    with tabs[1]:
        st.header("Thermal Design")
        metrics = [
            ("Q from hot stream", f"{Q_hot/1e6:.3f} MW"),
            ("Cold-water mass flow", f"{mc:.2f} kg/s"),
            ("ΔT1 = Th,in − Tc,out", f"{dT1:.2f} °C"),
            ("ΔT2 = Th,out − Tc,in", f"{dT2:.2f} °C"),
            ("Counter-current LMTD", f"{lmtd:.2f} °C"),
            ("F correction factor", f"{F_used:.3f}"),
            ("Corrected LMTD", f"{DT_eff:.2f} °C"),
            ("Preliminary area at U=350", f"{A_prelim:.1f} m²"),
        ]
        st.dataframe(pd.DataFrame(metrics, columns=["Item", "Value"]), use_container_width=True)
        st.latex(r"Q = \dot{m}_h C_{p,h}(T_{h,in}-T_{h,out})")
        st.latex(r"\Delta T_{lm}=\frac{\Delta T_1-\Delta T_2}{\ln(\Delta T_1/\Delta T_2)}")
        st.latex(r"A=\frac{Q}{U_o F \Delta T_{lm}}")
        if F_used < 0.75:
            st.warning("F factor is low. A different exchanger arrangement or more shell passes may be needed.")
    with tabs[2]:
        st.header("Kern Method Results")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tube-side h", f"{hi:.0f} W/m².K")
        c2.metric("Shell-side h", f"{ho_kern:.0f} W/m².K")
        c3.metric("Overall Uo", f"{Uo_kern:.0f} W/m².K")
        c4.metric("Required Area", f"{A_kern:.1f} m²")
        kern_table = pd.DataFrame({
            "Parameter": ["Tube Reynolds", "Tube Prandtl", "Tube velocity", "Tube pressure drop", "Shell equivalent diameter", "Shell Reynolds", "Shell Prandtl", "Shell-side mass velocity", "Number of baffles", "Shell pressure drop", "Required tubes", "Estimated shell diameter"],
            "Value": [f"{Re_t:,.0f}", f"{Pr_t:.2f}", f"{vt:.2f} m/s", f"{dp_tube/1000:.1f} kPa", f"{de*1000:.1f} mm", f"{Re_s:,.0f}", f"{Pr_s:.2f}", f"{Gs:.1f} kg/m².s", f"{Nb}", f"{dp_shell_kern/1000:.1f} kPa", f"{Nt_kern}", f"{Ds_k:.2f} m"]
        })
        st.dataframe(kern_table, use_container_width=True)
        st.subheader("Design checks")
        tube_dp_kPa = dp_tube / 1000
        shell_dp_kPa = dp_shell_kern / 1000
        checks = pd.DataFrame({
          "Check": [
            "Tube velocity typical target",
            "Tube Reynolds turbulent",
            "shell Reynolds turbulent",
            "F correction factor",
            "Tube pressure drop",
            "Shell pressure drop"
          ],
          "Result": [
            "✅ Acceptable" if 0.5 <= vt <= 3.0 else "⚠️ Check design",
            "✅ Acceptable" if Re_t >= 2300 else "⚠️ Check design",
            "✅ Acceptable" if Re_s >= 2300 else "⚠️ Check design",
            "✅ Acceptable" if F_used >= 0.75 else "⚠️ Check design",
            "✅ Acceptable" if tube_dp_kPa <= dp_allow_tube else "⚠️ Check design",
            "✅ Acceptable" if shell_dp_kPa <= dp_allow_shell else "⚠️ Check design"
          ]
        })
        st.table(checks)
    with tabs[3]:
        st.header("Delaware Method / Bell-Delaware Correction Framework")
        st.caption("This dashboard uses a Bell Delaware correction framework with key shell-side correction factors included, providing a realistic yet simplified representation of industrial heat exchanger behavior.")
        st.subheader("Delaware Flow Areas")
        flow_area_df = pd.DataFrame({
            "Flow Area": [
            "Cross-flow area, Sm",
            "Tube-to-baffle leakage area, Stb",
            "Shell-to-baffle leakage area, Ssb",
            "Bundle bypass area, Sb",
            "Window flow area, Sw"
            ],
        "Value (m²)": [Sm, Stb, Ssb, Sb, Sw]
        })
        st.dataframe(flow_area_df, use_container_width=True)
        st.subheader("Delaware Correction Factors")
        j_df = pd.DataFrame({
            "Correction Factor": [
            "Jc - baffle/window correction",
            "Jl - leakage correction",
            "Jb - bypass correction",
            "Js - baffle spacing correction",
            "Jr - laminar correction",
            "Total J product"
            ],
        "Value": [Jc, Jl, Jb, Js, Jr, J_total]
        })
        st.dataframe(j_df, use_container_width=True)
        st.subheader("Delaware Pressure Drop Breakdown")
        dp_delaware_df = pd.DataFrame({
        "Pressure Drop Region": [
            "Cross-flow pressure drop, ΔPc",
            "Window-flow pressure drop, ΔPw",
            "Entrance/exit pressure drop, ΔPe",
            "Total Delaware shell-side ΔP"
            ],
        "Value (kPa)": [
            dp_cross_delaware,
            dp_window_delaware,
            dp_end_delaware,
            dp_shell_delaware
            ]
        })
        st.dataframe(dp_delaware_df, use_container_width=True)
        st.subheader("Pressure Drop Correction Factors")
        r_df = pd.DataFrame({
        "Correction Factor": [
            "Rl - leakage pressure-drop correction",
            "Rb - bypass pressure-drop correction",
            "Rs - unequal spacing pressure-drop correction"
            ],
        "Value": [Rl, Rb, Rs]
        })
        st.dataframe(r_df, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Kern Shell ΔP", f"{dp_shell_kern/1000:.2f} kPa")
        c2.metric("Final Delaware ΔPf", f"{dp_shell_delaware:.2f} kPa")
        if dp_shell_delaware > 50:
         st.error("⚠️ Delaware shell-side pressure drop is too high. Increase shell diameter or baffle spacing.")
        else:
         st.success("✅ Delaware shell-side pressure drop is acceptable.")
        st.metric("Full Delaware shell-side h", f"{ho_delaware_full:.0f} W/m².K")
        st.metric("Full Delaware Overall Uo", f"{Uo_delaware_full:.0f} W/m².K")
        st.metric("Full Delaware Required Area", f"{A_delaware_full:.1f} m²")
        if J_total < 0.5:
         st.error("⚠️ Total Delaware correction factor is below 0.5 → poor shell-side design.")
        else:
         st.success("✅ Total Delaware correction factor is acceptable.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ideal h", f"{ho_kern:.0f} W/m².K")
        c2.metric("Total J correction", f"{J_total:.3f}")
        c3.metric("Corrected shell h", f"{ho_delaware:.0f} W/m².K")
        c4.metric("Overall Uo", f"{Uo_delaware:.0f} W/m².K")
        st.latex(r"h_{s,Delaware} = h_{ideal} \cdot J_c \cdot J_l \cdot J_b \cdot J_r \cdot J_s")
        del_table = pd.DataFrame({
            "Parameter": ["Delaware shell h", "Delaware overall Uo", "Delaware required area", "Delaware required tubes", "Estimated shell pressure drop"],
            "Value": [f"{ho_delaware:.0f} W/m².K", f"{Uo_delaware:.0f} W/m².K", f"{A_delaware:.1f} m²", f"{Nt_delaware}", f"{dp_shell_delaware:.1f} kPa"]
        })
        st.dataframe(del_table, use_container_width=True)
    with tabs[4]:
        st.header("🔁 Iteration & Optimization")
        st.markdown("### Final Converged Design")
        st.metric("Final Number of Tubes", f"{Nt:.0f}")
        st.metric("Final Shell Diameter", f"{Ds:.2f} m")
        st.metric("Required Area", f"{A_required:.2f} m²")
        st.metric("Overall U (final)", f"{Uo:.1f} W/m².K")
        st.success(f"Converged in {i+1} iterations")
        st.markdown("---")
        st.markdown("### Design Interpretation")
        if i < 5:
         st.success("Very fast convergence → excellent initial design 👍")
        elif i < 10:
         st.info("Moderate convergence → good design")
        else:
         st.warning("Slow convergence → consider improving initial guesses")
        st.markdown("## 🔍 Automatic Feasible Design Search")
        if st.button("Find Feasible Design", key="feasible_btn"):
          feasible_designs = []
          all_designs = []
          for Ds_try in np.linspace(0.3, 2.0, 25):   # wider shell sizes
              for B_ratio in np.linspace(0.2, 1.0, 20):  # wider spacing
              # try baffle spacing
                B_try = B_ratio * Ds_try
                # estimate tubes
                Nt_try = int(tube_count_from_area(A_required, do, L) * Nt_factor)
                # ---- Tube side ----
                tube_A, vt_try, Re_t_try, Pr_t_try, Nu_t_try, hi_try, ft_try, dp_tube_try = tube_side_calcs(
                    mc, rhoc, muc, cpc, kc, ndi, do, L, Nt_try, passes
                )
                # ---- Shell side ----
                de, As, Gs, Re_s_try, Pr_s, Nu_s, ho_try = kern_shell_h(
                    mh, cph, muh, kh, do, Ds_try, pitch, B_try, layout
                )
                Nb, dp_shell_kern_try = shell_dp_kern(Gs, rhoh, de, Ds_try, B_try)
                # Delaware correction and pressure drop from slide-based framework
                pd_try = delaware_pressure_drop(
                   mh, rhoh, Gs, Re_s_try, Ds_try, B_try, do, pitch, Nt_try,
                   baffle_cut, layout, sealing_strips, Nb
                )
                Rl_try = pd_try["Rl"]
                Rb_try = pd_try["Rb"]
                Rs_try = pd_try["Rs"]
                dp_ref_try = delaware_pressure_drop_reference(
                    Ds=Ds_try,
                    Db=Ds_try - estimate_clearance(Ds_try),
                    B=B_try,
                    baffle_cut=baffle_cut,
                    pitch=pitch,
                    do=do,
                    Nt=Nt_try,
                    Gs=Gs,
                    rho=rhoh,
                    Re_s=Re_s_try,
                    layout=layout,
                    Nb=Nb,
                    Rl=Rl_try,
                    Rb=Rb_try,
                    Rs=Rs_try
                )
                dp_shell_try = dp_ref_try["dp_f"]
                corr_try = delaware_heat_transfer_corrections(pd_try, Re_s_try, Nb)
                ho_del_try = ho_try * corr_try["Jtotal"]
                Uo_try = overall_U(hi_try, ho_del_try, ndi, do, kwall, Rfi, Rfo)
                dp_tube_try_kpa = dp_tube_try / 1000.0
                tube_dp_margin = dp_allow_tube - dp_tube_try_kpa
                shell_dp_margin = dp_allow_shell - dp_shell_try
                score = (
                    max(0, dp_tube_try_kpa - dp_allow_tube)
                    + max(0, dp_shell_try - dp_allow_shell)
                    + max(0, 0.5 - vt_try) * 100
                    + max(0, vt_try - 3) * 100
                    + max(0, 4000 - Re_t_try) / 100
                )
                all_designs.append({
                    "Ds (m)": Ds_try,
                    "B (m)": B_try,
                    "Nt": Nt_try,
                    "Tube velocity (m/s)": vt_try,
                    "Tube ΔP (kPa)": dp_tube_try_kpa,
                    "Shell ΔP (kPa)": dp_shell_try,
                    "Tube Re": Re_t_try,
                    "Uo (W/m².K)": Uo_try,
                    "Score": score
                })
                # Pressure drop (Delaware approx)
                # ---- CHECKS ----
                if (
                    0.5 <= vt_try <= 3 and
                    dp_tube_try_kpa < dp_allow_tube and
                    dp_shell_try < dp_allow_shell and
                    Re_t_try > 4000
                ):
                    feasible_designs.append({
                        "Ds": Ds_try,
                        "B": B_try,
                        "Nt": Nt_try,
                        "vt": vt_try,
                        "dp_tube": dp_tube_try_kpa,
                        "dp_shell": dp_shell_try,
                        "Uo": Uo_try,
                        "Area": A_required
                    })
          # ---- OUTPUT ----
          if len(feasible_designs) == 0:
             st.error("❌ No feasible design found. Try relaxing constraints.")
             st.info("""
             A feasible design is found only when:
             - Tube velocity is between 0.5 and 3 m/s
             - Tube-side pressure drop is below the allowable limit
             - Shell-side pressure drop is below the allowable limit
             - Tube Reynolds number is turbulent (Re > 4000)
             If no design is found, increase allowable ΔP, increase shell diameter search range, or allow more tube-count options.
             """)
             closest_df = pd.DataFrame(all_designs).sort_values("Score").head(5)
             st.subheader("Closest Designs Found")
             st.dataframe(closest_df, use_container_width=True)
          else:
             best = min(feasible_designs, key=lambda x: x["Area"])  # smallest area
             st.success("✅ Feasible design found!")
             st.write(f"**Nt:** {best['Nt']}")
             st.write(f"**Shell diameter Ds:** {best['Ds']:.2f} m")
             st.write(f"**Baffle spacing B:** {best['B']:.2f} m")
             st.write(f"**Tube velocity:** {best['vt']:.2f} m/s")
             st.write(f"**Tube ΔP:** {best['dp_tube']:.2f} kPa")
             st.write(f"**Shell ΔP:** {best['dp_shell']:.2f} kPa")
             st.write(f"**Overall U:** {best['Uo']:.1f} W/m².K")
    with tabs[5]:
        st.header("Kern vs Delaware Comparison")
        comp = pd.DataFrame({
            "Result": ["Shell-side h", "Overall Uo", "Required area", "Required tubes", "Shell pressure drop"],
            "Kern": [ho_kern, Uo_kern, A_kern, Nt_kern, dp_shell_kern/1000],
            "Delaware": [ho_delaware, Uo_delaware, A_delaware, Nt_delaware, dp_shell_delaware]
        })
        st.dataframe(comp, use_container_width=True)
        plot_df = comp.melt(id_vars="Result", var_name="Method", value_name="Value")
        fig = px.bar(plot_df, x="Result", y="Value", color="Method", barmode="group", title="Method Comparison")
        st.plotly_chart(fig, use_container_width=True)
        st.success("Engineering interpretation: Delaware normally predicts a lower effective shell-side coefficient because it penalizes ideal crossflow for leakage, bypassing, and window effects. Therefore, Delaware often gives a larger required area than Kern.")
    with tabs[6]:
        st.header("Sensitivity Analysis")
        st.caption("Choose a variable to test how design results change.")
        analysis = st.selectbox(
        "Sensitivity variable",
        ["Baffle spacing ratio B/Ds", "Tube outside diameter", "Hot-fluid mass flow","Shell-side fouling resistance"]
        )
        rows = []
        if analysis == "Baffle spacing ratio B/Ds":
            values = np.linspace(0.2, 1.0, 25)
            for br in values:
                Bx = br * Ds
                de_x, As_x, Gs_x, Re_s_x, Pr_s_x, Nu_s_x, ho_x = kern_shell_h(
                mh, cph, muh, kh, do, Ds, pitch, Bx, layout
                )
                Ux = overall_U(hi, ho_x * corr["Jtotal"], ndi, do, kwall, Rfi, Rfo)
                Ax = safe_div(Q_hot, Ux * DT_eff)
                rows.append([br, ho_x, Ux, Ax, Re_s_x])
            df = pd.DataFrame(
               rows,
               columns=["B/Ds", "Shell h", "Overall Uo", "Area", "Shell Reynolds"]
            )
            ycol = st.selectbox(
                "Y-axis",
                ["Shell h", "Overall Uo", "Area", "Shell Reynolds"]
            )
            fig = px.line(
                df,
                x="B/Ds",
                y=ycol,
                markers=True,
                title=f"Effect of baffle spacing on {ycol}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        elif analysis == "Tube outside diameter":
            values = np.linspace(0.0127, 0.03175, 25)
            for dox in values:
                dix = dox * 0.84
                pitchx = pitch_ratio * dox
                Ntx = tube_count_from_area(A_delaware * (1 + design_margin / 100), dox, L)
                Dbx, Dsx = shell_diameter_estimate(Ntx, pitchx, passes, layout)
                tube_Ax, vt_x, Re_t_x, Pr_t_x, Nu_t_x, hi_x, ft_x, dp_tube_x = tube_side_calcs(
                    mc, rhoc, muc, cpc, kc, dix, dox, L, Ntx, passes
                )
                de_x, As_x, Gs_x, Re_s_x, Pr_s_x, Nu_s_x, ho_x = kern_shell_h(
                    mh, cph, muh, kh, dox, Dsx, pitchx, B, layout
                )
                Ux = overall_U(hi_x, ho_x * corr["Jtotal"], dix, dox, kwall, Rfi, Rfo)
                Ax = safe_div(Q_hot, Ux * DT_eff)
                rows.append([dox * 1000, Ux, Ax, Ntx, Dsx, vt_x, dp_tube_x])
            df = pd.DataFrame(
                rows,
                columns=["Tube OD mm", "Overall Uo", "Area", "Number of tubes", "Shell diameter", "Tube velocity", "Tube ΔP"]
            )
            ycol = st.selectbox(
                "Y-axis",
                ["Overall Uo", "Area", "Number of tubes", "Shell diameter", "Tube velocity", "Tube ΔP"]
            )
            fig = px.line(
                df,
                x="Tube OD mm",
                y=ycol,
                markers=True,
                title=f"Effect of tube diameter on {ycol}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
        else:
            values = np.linspace(max(1, 0.5 * mh), 1.8 * mh, 25)
            for mhx in values:
                Qx = mhx * cph * (Th_in - Th_out)
                mcx = safe_div(Qx, cpc * (Tc_out - Tc_in))
                tube_Ax, vt_x, Re_t_x, Pr_t_x, Nu_t_x, hi_x, ft_x, dp_tube_x = tube_side_calcs(
                    mcx, rhoc, muc, cpc, kc, ndi, do, L, Nt, passes
                )
                de_x, As_x, Gs_x, Re_s_x, Pr_s_x, Nu_s_x, ho_x = kern_shell_h(
                    mhx, cph, muh, kh, do, Ds, pitch, B, layout
                )
                Ux = overall_U(hi_x, ho_x * corr["Jtotal"], ndi, do, kwall, Rfi, Rfo)
                Ax = safe_div(Qx, Ux * DT_eff)
                rows.append([mhx, Qx / 1e6, Re_s_x, Ux, Ax])
            df = pd.DataFrame(
                rows,
                columns=["Hot mass flow kg/s", "Duty MW", "Shell Reynolds", "Overall Uo", "Area"]
            )
            ycol = st.selectbox(
                "Y-axis",
                ["Duty MW", "Shell Reynolds", "Overall Uo", "Area"]
            )
            fig = px.line(
                df,
                x="Hot mass flow kg/s",
                y=ycol,
                markers=True,
                title=f"Effect of hot-fluid flow rate on {ycol}"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df, use_container_width=True)
    with tabs[7]:
        st.header("Design Validation & Checks")
        method_check = st.radio(
        "Select method for shell-side checks:",
        ["Delaware", "Kern"]
        )
        st.subheader("Tube Side Checks")
        if vt > 3:
            st.error(f"⚠️ Tube velocity too HIGH: {vt:.2f} m/s (Risk of erosion)")
        elif vt < 0.5:
            st.warning(f"⚠️ Tube velocity too LOW: {vt:.2f} m/s (Poor heat transfer)")
        else:
            st.success(f"✅ Tube velocity OK: {vt:.2f} m/s")
        if Re_t < 2300:
            st.warning(f"⚠️ Laminar flow in tubes (Re = {Re_t:.0f}) → Low heat transfer")
        else:
            st.success(f"✅ Turbulent flow in tubes (Re = {Re_t:.0f})")
        st.subheader("Shell Side Checks")
        if Re_s < 2000:
            st.warning(f"⚠️ Shell flow may be laminar (Re = {Re_s:.0f})")
        else:
            st.success(f"✅ Shell flow turbulent (Re = {Re_s:.0f})")
        st.subheader("Pressure Drop Checks")
        dp_tube_kPa = dp_tube / 1000
        if dp_tube_kPa > dp_allow_tube:
            st.warning(f"⚠️ Tube pressure drop HIGH: {dp_tube_kPa:.2f} kPa (Limit = {dp_allow_tube:.2f} kPa)")
        else:
            st.success(f"✅ Tube pressure drop OK: {dp_tube_kPa:.2f} kPa")
        # Select method for shell-side check
        if method_check == "Delaware":
            dp_shell_kPa = dp_shell_delaware
        else:
            dp_shell_kPa = dp_shell_kern / 1000  # convert Pa → kPa
        if dp_shell_kPa > dp_allow_shell:
            st.warning(f"⚠️ Shell pressure drop HIGH: {dp_shell_kPa:.2f} kPa (Limit = {dp_allow_shell:.2f} kPa)")
        else:
            st.success(f"✅ Shell pressure drop OK: {dp_shell_kPa:.2f} kPa")
        st.subheader("Automatic Design Recommendations")
    recommendations = []
    dp_tube_kPa = dp_tube / 1000
    if dp_tube_kPa > dp_allow_tube:
        recommendations.append(
            f"Tube-side pressure drop is too high: {dp_tube_kPa:.2f} kPa "
            f"(limit = {dp_allow_tube:.2f} kPa). Increase tube diameter, increase number of tubes, "
            "reduce tube length, or reduce tube passes."
        )
    if dp_shell_kPa > dp_allow_shell:
        recommendations.append(
            f"Shell-side pressure drop is too high: {dp_shell_kPa:.2f} kPa "
            f"(limit = {dp_allow_shell:.2f} kPa). Increase shell diameter, increase baffle spacing, "
            "or reduce shell-side velocity."
        )
    if vt > 3:
        recommendations.append("Tube velocity is high. Increase tube diameter or increase number of tubes.")
    if vt < 0.5:
        recommendations.append("Tube velocity is low. Reduce tube diameter or reduce number of tubes to improve heat transfer.")
    if Re_t < 2300:
        recommendations.append("Tube-side flow is laminar. Increase flow rate or reduce flow area.")
    if Re_s < 2000:
        recommendations.append("Shell-side flow may be weak. Reduce baffle spacing or increase shell-side velocity.")
    if len(recommendations) == 0:
        st.success("✅ No major design issues detected.")
    else:
        for i, rec in enumerate(recommendations, start=1):
            st.info(f"Recommendation {i}: {rec}")
    st.subheader("Suggested Design Actions")
    if dp_tube_kPa > dp_allow_tube:
        st.write("🔧 Increase tube diameter OR increase number of tubes OR reduce tube length/passes to lower tube-side pressure drop.")
    if dp_shell_kPa > dp_allow_shell:
        st.write("🔧 Increase shell diameter (Ds) OR increase baffle spacing (B) to reduce shell-side pressure drop.")
    if vt > 3:
        st.write("⚠️ Tube velocity too high → risk of erosion. Increase tube diameter or number of tubes")
    if vt < 0.5:
        st.write("⚠️ Tube velocity too low → poor heat transfer. Reduce diameter or reduce number of tubes")
    with tabs[8]:
        st.markdown("""
        <style>
        .report-card {
            background-color: rgba(248, 251, 255, 0.95);
            border: 2px solid #5d81a3;
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 20px;
            box-shadow: 0px 5px 10px rgba(0,0,0,0.15);
            color: #000;
        }
        .report-header {
            text-align: center;
            font-size: 22px;
            font-weight: 900;
            margin-bottom: 15px;
            color: #1a1a1a;
            text-transform: uppercase;
        }
        .col-title {
            font-weight: bold;
            font-size: 18px;
            text-align: center;
            margin-bottom: 10px;
            color: #000;
        }
        .metric-title {
            text-align: center;
            font-size: 17px;
            color: #333;
        }
        .metric-value {
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            color: #000;
            margin-top: 5px;
        }
        .rating {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin-top: 5px;
        }
        .comp-table { width: 100%; font-size: 16px; border-collapse: collapse; color: #000; }
        .comp-table td { padding: 4px; border: none; }
        .comp-table .val { text-align: right; font-weight: bold; }
        
        .export-btn {
            display: block; width: 100%; padding: 12px; margin-bottom: 12px;
            text-align: center; border-radius: 8px; font-weight: bold;
            color: white !important; text-decoration: none; font-size: 16px;
        }
        .btn-pdf { background-color: #4a7bb0; }
        .btn-excel { background-color: #3e8e41; }
        </style>
        """, unsafe_allow_html=True)

        # 1. RESULT METRICS
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-header">RESULT METRICS</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="metric-title">Heat Transfer Coefficient (U)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{Uo_delaware:.2f} W/m².K</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-title">Pressure Drop ΔP</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{dp_shell_delaware:.2f} kPa</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-title">Overall Performance Rating:</div>', unsafe_allow_html=True)
            if dp_shell_delaware <= dp_allow_shell and (dp_tube/1000) <= dp_allow_tube:
                rating_html = '<span style="color:#e6a817;">Acceptable</span>/<span style="color:#4CAF50;">Optimal</span>'
            else:
                rating_html = '<span style="color:red;">Needs Review</span>'
            st.markdown(f'<div class="rating">{rating_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 2. METHOD COMPARISON
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-header">METHOD COMPARISON</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="col-title">Kern Method</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="comp-table">
                <tr><td>Kern Method U</td><td class="val">{Uo_kern:.1f} W/m².K</td></tr>
                <tr><td>Shell-side h</td><td class="val">{ho_kern:.1f} W/m².K</td></tr>
                <tr><td>Required Area</td><td class="val">{A_kern:.2f} m²</td></tr>
                <tr><td>Pressure Drop (ΔP)</td><td class="val">{dp_shell_kern/1000:.2f} kPa</td></tr>
                <tr><td>Number of Tubes</td><td class="val">{Nt_kern}</td></tr>
                <tr><td>Estimated Ds</td><td class="val">{shell_diameter_estimate(Nt_kern, pitch, passes, layout)[1]:.2f} m</td></tr>
            </table>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="col-title">Delaware Method</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <table class="comp-table">
                <tr><td>Delaware Method U</td><td class="val">{Uo_delaware:.1f} W/m².K</td></tr>
                <tr><td>Shell-side h</td><td class="val">{ho_delaware:.1f} W/m².K</td></tr>
                <tr><td>Required Area</td><td class="val">{A_delaware:.2f} m²</td></tr>
                <tr><td>Pressure Drop (ΔP)</td><td class="val">{dp_shell_delaware:.2f} kPa</td></tr>
                <tr><td>Number of Tubes</td><td class="val">{Nt_delaware}</td></tr>
                <tr><td>Estimated Ds</td><td class="val">{shell_diameter_estimate(Nt_delaware, pitch, passes, layout)[1]:.2f} m</td></tr>
            </table>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. PERFORMANCE CHARTS
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-header">PERFORMANCE CHARTS</div>', unsafe_allow_html=True)
        c_chart1, c_chart2 = st.columns(2)
        with c_chart1:
            plot_m = np.linspace(mh*0.5, mh*1.5, 10)
            plot_U = []
            for m_test in plot_m:
                _, _, _, _, _, _, ho_test = kern_shell_h(m_test, cph, muh, kh, do, Ds, pitch, B, layout)
                Ux = overall_U(hi, ho_test * corr["Jtotal"], ndi, do, kwall, Rfi, Rfo)
                plot_U.append(Ux)
            fig1 = px.line(x=plot_m, y=plot_U, markers=True, labels={'x': 'Flow Rate (kg/s)', 'y': 'U (W/m².K)'})
            fig1.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='black', size=14))
            fig1.update_xaxes(color='black', gridcolor='rgba(0,0,0,0.1)', showline=True, linewidth=1, linecolor='black')
            fig1.update_yaxes(color='black', gridcolor='rgba(0,0,0,0.1)', showline=True, linewidth=1, linecolor='black')
            st.plotly_chart(fig1, use_container_width=True, theme=None)
        with c_chart2:
            plot_B = np.linspace(0.2, 0.8, 6) * Ds
            plot_DP = []
            for B_test in plot_B:
                Nb_test, _ = shell_dp_kern(Gs, rhoh, de, Ds, B_test)
                pd_test = delaware_pressure_drop(mh, rhoh, Gs, Re_s, Ds, B_test, do, pitch, Nt, baffle_cut, layout, sealing_strips, Nb_test)
                plot_DP.append(pd_test["dp_total"])
            fig2 = px.bar(x=[f"{b*1000:.0f} mm" for b in plot_B], y=plot_DP, labels={'x': 'Shell Side Geometry (Baffle Spacing)', 'y': 'ΔP (kPa)'})
            fig2.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color='black', size=14))
            fig2.update_xaxes(color='black', gridcolor='rgba(0,0,0,0.1)', showline=True, linewidth=1, linecolor='black')
            fig2.update_yaxes(color='black', gridcolor='rgba(0,0,0,0.1)', showline=True, linewidth=1, linecolor='black')
            fig2.update_traces(marker_color='#5d81a3')
            st.plotly_chart(fig2, use_container_width=True, theme=None)
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. RECOMMENDATIONS
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown('<div class="report-header">ENGINEERING RECOMMENDATIONS</div>', unsafe_allow_html=True)
        recs_html = '<ul style="font-size: 17px; line-height: 1.6; color: #000;">'
        if (dp_tube/1000) > dp_allow_tube:
            recs_html += "<li>Consider increasing tube diameter or number of tubes to reduce tube-side pressure drop.</li>"
        if dp_shell_delaware > dp_allow_shell:
            recs_html += "<li>Consider increasing baffle spacing or shell diameter to reduce shell-side pressure drop.</li>"
        if vt > 3:
            recs_html += "<li>Tube velocity is high. Consider increasing flow area to reduce erosion risk.</li>"
        if "<li>" not in recs_html:
            recs_html += "<li>Consider increasing tube length to reduce pressure drop.</li>"
            recs_html += "<li>Review shell-side baffle spacing.</li>"
            recs_html += "<li>Ensure mechanical constraints align with the calculated geometry.</li>"
        recs_html += "</ul>"
        st.markdown(recs_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
