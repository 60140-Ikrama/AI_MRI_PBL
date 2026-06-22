---
name: Clinical Precision AI
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#bcc9cd'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#869397'
  outline-variant: '#3d494c'
  surface-tint: '#4cd7f6'
  primary: '#4cd7f6'
  on-primary: '#003640'
  primary-container: '#06b6d4'
  on-primary-container: '#00424f'
  inverse-primary: '#00687a'
  secondary: '#ddb7ff'
  on-secondary: '#490080'
  secondary-container: '#6f00be'
  on-secondary-container: '#d6a9ff'
  tertiary: '#c4c7c9'
  on-tertiary: '#2d3133'
  tertiary-container: '#a4a7a9'
  on-tertiary-container: '#393d3e'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#acedff'
  primary-fixed-dim: '#4cd7f6'
  on-primary-fixed: '#001f26'
  on-primary-fixed-variant: '#004e5c'
  secondary-fixed: '#f0dbff'
  secondary-fixed-dim: '#ddb7ff'
  on-secondary-fixed: '#2c0051'
  on-secondary-fixed-variant: '#6900b3'
  tertiary-fixed: '#e0e3e5'
  tertiary-fixed-dim: '#c4c7c9'
  on-tertiary-fixed: '#191c1e'
  on-tertiary-fixed-variant: '#444749'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  data-mono:
    fontFamily: Geist
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-caps:
    fontFamily: Geist
    fontSize: 11px
    fontWeight: '700'
    lineHeight: 12px
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
  container-padding: 16px
  gutter: 12px
  component-gap: 8px
  density-high: 4px
  density-standard: 12px
---

## Brand & Style

The design system is engineered for high-stakes medical research and diagnostic workflows. It balances industrial-grade reliability with cutting-edge computational intelligence. The personality is authoritative, clinical, and sophisticated, designed to instill trust in AI-driven insights while maintaining the rigor required for medical environments.

The visual style is **Corporate Modern with Glassmorphic accents**. It utilizes high-density layouts to accommodate complex data sets and multi-modal imaging. Subtle frosted glass effects are reserved for diagnostic overlays and floating research tools, ensuring that the primary focus remains on the medical imagery. The aesthetic is clean, precise, and technologically advanced, evoking the feel of a modern medical workstation.

## Colors

The palette is optimized for long-duration research sessions and high-contrast diagnostic viewing.

- **Primary (Cyan):** Used for active data points, selection states, and scanning progress. It represents the "human-in-the-loop" interaction.
- **Secondary (Purple):** Reserved exclusively for AI-generated insights, machine learning confidence scores, and automated segmentation.
- **Surface Tiers:** 
  - **Base:** Deep Charcoal (#0F172A) for the primary workspace.
  - **Elevated:** Slate (#1E293B) for sidebars, toolbars, and card backgrounds.
- **Clinical Light Mode:** For documentation and reporting, use a Medical White (#F8FAFC) base with Deep Navy text to maintain professional authority.
- **Semantic Colors:** Critical alerts use a high-visibility Red, Stable metrics use Green, and Processing/Warning states use Amber.

## Typography

Typography focuses on legibility and information density. **Inter** serves as the primary typeface for its exceptional clarity in UI contexts. For technical data, coordinates, and AI confidence intervals, **Geist** (a technical mono-influenced sans) is used to provide a distinct visual "texture" for machine-readable information.

- **Headlines:** Use tight tracking and semi-bold weights to establish clear hierarchy without wasting vertical space.
- **Data Mono:** Specifically for telemetry, coordinates, and pixel values.
- **Label Caps:** Used for metadata headers and secondary categorization within high-density cards.

## Layout & Spacing

The layout employs a **Fluid Grid** model optimized for high-density medical workstations. It utilizes a 4px baseline shift to maintain tight alignment.

- **Research Dashboard:** A 12-column layout on desktop with minimal margins (16px) to maximize the "viewport" for medical images.
- **Medical Workstation Density:** Spacing between related data points is kept at 4px or 8px. Cards use 12px internal padding to ensure data is legible but compact.
- **Imaging Viewports:** These bypass standard grid constraints to allow for flexible 1-up, 2-up, or 4-up synchronized viewing panels.
- **Responsive Behavior:** On tablet, the layout reflows to an 8-column grid; mobile viewports prioritize a single vertical data stream with collapsed toolbars.

## Elevation & Depth

This design system uses **Tonal Layers** combined with **Glassmorphism** for its elevation model.

1.  **Level 0 (Base):** Deep Charcoal (#0F172A). Background for the entire application.
2.  **Level 1 (Surface):** Slate (#1E293B). Used for secondary panels and the main container background. No shadows; separated by 1px borders in a lighter slate (#334155).
3.  **Level 2 (Diagnostic Overlays):** Semi-transparent glass panels (Background blur 12px, 60% opacity of Slate). These sit above imaging data for measurement tools and ROI (Region of Interest) callouts.
4.  **Level 3 (Modals/Popovers):** Subtle ambient shadows (0 10px 25px -5px rgba(0,0,0,0.5)) with a 1px border.

Shadows are avoided for standard UI components to keep the interface feeling flat and "industrial." Depth is primarily communicated through color shifts and 1px borders.

## Shapes

The shape language is **Soft (0.25rem)**. This provides a professional, "machined" aesthetic that avoids the playfulness of larger radiuses while being more modern and accessible than sharp 90-degree corners.

- **Cards & Primary Containers:** 4px (0.25rem) radius.
- **Large Action Buttons:** 4px (0.25rem) radius.
- **Status Badges:** Fully rounded (pill-shaped) to differentiate them from interactive buttons.
- **Input Fields:** 4px (0.25rem) radius with 1px solid borders.

## Components

- **High-Density Data Cards:** Use Level 1 surfaces with Level 0 headers. Group data with `label-caps` typography and `data-mono` values.
- **Medical Imaging Viewport:** Black background (#000000). Features a 1px Cyan border when active. Overlay tools use glassmorphism with white icons.
- **Buttons:**
    - **Primary:** Solid Cyan with dark text. 
    - **AI Insight:** Solid Purple with white text.
    - **Secondary/Ghost:** Slate border with Cyan text for low-priority actions.
- **Status Badges:** Small, high-contrast pills. 
    - *Critical:* Red background, white text.
    - *Stable:* Transparent with Green border and Green dot indicator.
- **Input Fields:** Dark slate background with a 1px border that glows Cyan on focus.
- **Medical Charts:** Minimalist line and bar charts using primary and secondary colors. No grid lines; use "Geist" for axis labels.
- **Progress Steppers:** Use thin Cyan lines to indicate AI processing stages, with a Purple pulse animation for active ML inference.