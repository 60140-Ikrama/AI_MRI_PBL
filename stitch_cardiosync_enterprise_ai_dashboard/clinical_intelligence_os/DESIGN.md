---
name: Clinical Intelligence OS
colors:
  surface: '#131315'
  surface-dim: '#131315'
  surface-bright: '#39393b'
  surface-container-lowest: '#0e0e10'
  surface-container-low: '#1b1b1d'
  surface-container: '#1f1f21'
  surface-container-high: '#2a2a2c'
  surface-container-highest: '#343536'
  on-surface: '#e4e2e4'
  on-surface-variant: '#c5c6cd'
  inverse-surface: '#e4e2e4'
  inverse-on-surface: '#303032'
  outline: '#8f9097'
  outline-variant: '#44474d'
  surface-tint: '#b9c7e4'
  primary: '#b9c7e4'
  on-primary: '#233148'
  primary-container: '#0a192f'
  on-primary-container: '#74829d'
  inverse-primary: '#515f78'
  secondary: '#44d8f1'
  on-secondary: '#00363e'
  secondary-container: '#00bcd4'
  on-secondary-container: '#004650'
  tertiary: '#e7bf99'
  on-tertiary: '#432b10'
  tertiary-container: '#281400'
  on-tertiary-container: '#9d7b5a'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#d6e3ff'
  primary-fixed-dim: '#b9c7e4'
  on-primary-fixed: '#0d1c32'
  on-primary-fixed-variant: '#39475f'
  secondary-fixed: '#a1efff'
  secondary-fixed-dim: '#44d8f1'
  on-secondary-fixed: '#001f25'
  on-secondary-fixed-variant: '#004e59'
  tertiary-fixed: '#ffdcbd'
  tertiary-fixed-dim: '#e7bf99'
  on-tertiary-fixed: '#2b1701'
  on-tertiary-fixed-variant: '#5d4124'
  background: '#131315'
  on-background: '#e4e2e4'
  surface-variant: '#343536'
typography:
  display-xl:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  data-vitals:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: 0.05em
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-mono:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.08em
  status-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.03em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 8px
  sm: 16px
  md: 24px
  lg: 40px
  xl: 64px
  gutter: 20px
  margin-mobile: 16px
  margin-desktop: 32px
---

## Brand & Style

The design system is engineered for high-stakes hematology analysis, projecting an aura of absolute precision, clinical authority, and futuristic intelligence. It targets medical professionals who require rapid, accurate data interpretation in high-pressure environments. 

The aesthetic is a sophisticated blend of **Industrial Futurism** and **Modern Glassmorphism**. The interface prioritizes data density without compromising clarity, utilizing deep layers and translucent surfaces to organize complex hematological workflows. Every element is designed to minimize cognitive load and eye strain during long shifts, using high-contrast dark modes and subtle luminous accents to guide the user's focus to critical vitals and AI-driven insights.

## Colors

The palette is anchored in **Deep Clinical Navy**, providing a stable, low-fatigue background that recedes, allowing data to pop. 

- **Primary & Neutral:** The foundation uses layered shades of Navy to create visual hierarchy and depth.
- **Hematology Red:** Reserved strictly for critical alerts, active blood-related data, and life-threatening anomalies.
- **Digital Cyan:** Used for active analysis, scanning states, and general interactive elements to signify "logic."
- **Cyber Purple:** Dedicated to AI-assisted insights, machine learning confidence scores, and automated diagnostic suggestions.
- **Functional Colors:** Clinical Emerald and Amber Analytics provide standard status signaling for stable and cautionary data ranges, respectively.

## Typography

This design system utilizes **Inter** for its exceptional legibility in digital interfaces and **JetBrains Mono** for technical data labels to evoke a sense of precision and "instrumentation."

- **Data Vitals:** Large, bold numerical values use increased letter spacing to ensure no two digits are misread.
- **Monospaced Labels:** Technical metadata, timestamps, and laboratory references use JetBrains Mono to differentiate "system data" from "human interpretation."
- **Visual Hierarchy:** Use weight rather than color to establish hierarchy in text, preserving color for functional status indicators.

## Layout & Spacing

The layout follows a **Hybrid Fluid-Modular Grid**. While the main dashboard components are fluid to maximize screen real estate on medical-grade monitors, internal component spacing is strictly governed by a 4px baseline grid.

- **Desktop:** A 12-column grid with 20px gutters. Lateral panels for AI analysis should occupy a fixed 320px width, while the central workspace scales.
- **Tablet:** 8-column grid, shifting to vertical stacking for diagnostic waveforms.
- **Medical Context:** Content density is high. Use "SM" and "XS" spacing for internal card padding to allow as much data on screen as possible while maintaining a clean "clinical" look through precise alignment.

## Elevation & Depth

Depth is achieved through **Glassmorphism and Tonal Layering** rather than traditional drop shadows.

- **Surface 0 (Base):** Deep Clinical Navy (#0A192F).
- **Surface 1 (Cards):** Slightly lighter navy (#112240) with a 1px inner stroke of 10% Cyan to simulate a "beveled glass" edge.
- **Surface 2 (Overlays/Modals):** Semi-transparent background blur (20px) with a 40% opacity navy fill.
- **Luminous Highlights:** Use ultra-soft, colored outer glows (10-15px blur, 15% opacity) to indicate "Active" or "Critical" containers, using the respective status color (Cyan for active, Red for critical).

## Shapes

The shape language is "Soft-Industrial." While 0.5rem (8px) is the standard for cards and primary containers, certain elements follow specific rules:
- **Interactive Elements:** Buttons and toggles use the standard 8px radius.
- **Data Containers:** Waveform and cell containers utilize a "rounded-lg" (1rem) radius to feel more organic and clinical.
- **Status Pills:** Fully rounded (pill-shaped) to distinguish them from actionable buttons.

## Components

### Medical Cards
Cards feature a semi-transparent "glass" background with a subtle cyan border. Include a "Data Spark" — a miniature, simplified line chart in the top right corner — to show the trend of the specific vital (e.g., WBC count) over the last 24 hours.

### Circular Progress (Confidence Scores)
AI confidence is visualized through concentric rings. The outer ring represents the total confidence percentage, using Cyber Purple. The stroke should be "segmented" to look like a high-precision gauge.

### Cell & Waveform Containers
These containers must support high-fidelity visualization. The background is a darker, desaturated navy to maximize the contrast of the waveform. Add a 10% opacity grid overlay (4px squares) to the background to assist with manual ocular measurement.

### Buttons & Inputs
- **Primary Button:** Solid Digital Cyan with black text for maximum contrast.
- **AI Action:** Gradient fill (Cyber Purple to Clinical Navy) with a luminous purple "aura."
- **Inputs:** Ghost-style borders that illuminate to Digital Cyan on focus.

### Status Indicators
Small, glowing "LED" style pips next to text labels. Use a pulse animation for "Critical" (Red) and "Analyzing" (Cyan) states to indicate real-time processing.