---
name: CardiaPro Industrial
colors:
  surface: '#0e141c'
  surface-dim: '#0e141c'
  surface-bright: '#343a42'
  surface-container-lowest: '#090f16'
  surface-container-low: '#161c24'
  surface-container: '#1a2028'
  surface-container-high: '#242a33'
  surface-container-highest: '#2f353e'
  on-surface: '#dde3ee'
  on-surface-variant: '#bacbb9'
  inverse-surface: '#dde3ee'
  inverse-on-surface: '#2b3139'
  outline: '#859585'
  outline-variant: '#3b4a3d'
  surface-tint: '#00e475'
  primary: '#75ff9e'
  on-primary: '#003918'
  primary-container: '#00e676'
  on-primary-container: '#00612e'
  inverse-primary: '#006d35'
  secondary: '#81cfff'
  on-secondary: '#00344b'
  secondary-container: '#00a9e8'
  on-secondary-container: '#003952'
  tertiary: '#ebdfff'
  on-tertiary: '#342c49'
  tertiary-container: '#cfc2e7'
  on-tertiary-container: '#584e6e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#62ff96'
  primary-fixed-dim: '#00e475'
  on-primary-fixed: '#00210b'
  on-primary-fixed-variant: '#005226'
  secondary-fixed: '#c6e7ff'
  secondary-fixed-dim: '#81cfff'
  on-secondary-fixed: '#001e2d'
  on-secondary-fixed-variant: '#004c6b'
  tertiary-fixed: '#eaddff'
  tertiary-fixed-dim: '#cdc1e5'
  on-tertiary-fixed: '#1f1732'
  on-tertiary-fixed-variant: '#4b4260'
  background: '#0e141c'
  on-background: '#dde3ee'
  surface-variant: '#2f353e'
typography:
  vitals-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: '1'
    letterSpacing: -0.02em
  vitals-md:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1'
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  data-mono:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 16px
  margin-mobile: 12px
  margin-desktop: 24px
  data-density-gap: 8px
---

## Brand & Style
The design system is engineered for mission-critical healthcare environments where split-second decision-making is paramount. The brand personality is **authoritative, clinical, and high-fidelity**, mirroring the reliability of professional diagnostic equipment. It targets cardiologists, ER nurses, and technicians who operate in high-stress, variable-light conditions.

The aesthetic blends **Corporate Modernism** with **Functional Glassmorphism**. It utilizes semi-transparent overlays and subtle micro-glows to indicate active monitoring states without inducing visual fatigue. The interface prioritizes data density and "glanceability," ensuring that life-critical vitals are always the most prominent elements on the screen.

## Colors
The palette is rooted in functional utility. The default **Dark Cardiology** mode uses a #0A0F14 Deep Onyx base to reduce screen glare in dimmed clinical wards, while the **Clinical White** mode provides high-contrast legibility for bright hospital environments.

- **Primary (Neon Medical Green):** Reserved for stable vitals, active heart rates, and "safe" system states.
- **Alert & Warning:** Used sparingly for high-priority physiological alarms (Critical Red) and technical alerts (Amber).
- **AI/Insight:** A soft lavender used to distinguish machine-learning observations from raw physiological data.
- **Functional Gradients:** Subtle 15% opacity glows are applied to primary elements to simulate the phosphor-persistence of hardware monitors.

## Typography
The typography system uses a dual-font approach. **Inter** handles all UI labels and patient data for maximum legibility across viewing angles. **JetBrains Mono** is utilized for timecodes, waveform scales, and numeric vitals to prevent "character jump" as values fluctuate.

Vitals utilize a massive scale (`vitals-lg`) to ensure they are readable from a distance of 3-5 meters. Labels are consistently uppercase with increased tracking to differentiate them from dynamic data values.

## Layout & Spacing
This design system employs a **high-density fluid grid** based on a 4px baseline. In clinical dashboards, whitespace is minimized in favor of information density, using 16px gutters to separate distinct monitoring modules.

- **Mobile:** Single-column focus on the primary waveform with a bottom-docked "Vitals Strip" for tactile control.
- **Tablet/Desktop:** 12-column layout. Left 8 columns are dedicated to real-time waveform grids; right 4 columns are for historical trends and AI insights.
- **Waveform Grids:** Must maintain a strict aspect ratio. Backgrounds should feature a 10px/50px sub-grid pattern to assist in manual rhythm interpretation.

## Elevation & Depth
Depth is conveyed through **Tonal Layering** and **1px technical strokes**. 
- **Base Level:** Deep Onyx (#0A0F14).
- **Surface Level:** Slate Grey (#161C24) with a 1px solid border (#2C343F) to define module boundaries.
- **Overlays:** Glassmorphism is used for temporary modals and dropdowns, utilizing a `backdrop-filter: blur(12px)` and 10% white opacity tint.
- **Interactive States:** Instead of heavy shadows, active elements use a **micro-glow** (2px blur, color-matched to the primary or alert status) to signify focus or alarm state.

## Shapes
Shapes are **Soft (4px - 8px radius)**, providing a professional, engineered feel that avoids the "consumer" look of highly rounded corners.
- **Containers:** 4px radius for internal data cards.
- **Buttons:** 6px radius for tactile controls.
- **Status Indicators:** Circular (pill) for AI tags, while physiological alarms remain rectangular to maximize screen-edge visibility.

## Components
- **Vitals Cards:** High-contrast containers featuring a large numeric value, a "trend" sparkline, and an uppercase unit label (e.g., "BPM").
- **Clinical Buttons:** Large, tactile hit areas (min 48px) with inset 1px borders. Active states use a solid fill of the primary color with dark text.
- **Waveform Monitor:** A canvas-based component with a "phosphor" sweep-bar effect. The line thickness is 2pt with a subtle outer glow of the same color.
- **Alarm Banners:** Full-width top bars using the Alert (#FF5252) or Warning (#FFAB40) colors. Text must be high-contrast (White or Deep Onyx) and utilize the `label-caps` style.
- **Data Tables:** Compact rows with 1px dividers. Monospaced font for all numerical columns to ensure alignment.
- **Toggle Switches:** Designed as "Rockers" to mimic physical medical hardware, providing clear "On/Off" visual state.