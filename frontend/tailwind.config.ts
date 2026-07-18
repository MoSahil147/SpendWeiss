import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm paper ground and crisp card surfaces, deliberately not a cool gray.
        paper: "#F5F4EF",
        surface: {
          DEFAULT: "#FFFFFF",
          raise: "#FCFBF7",
        },
        // Deep teal-charcoal ink instead of a pure/neutral black.
        ink: "#14211D",
        muted: "#5C6B66",
        line: "#E5E1D6",
        // Peacock teal: the brand colour and the "this decision happened" colour.
        teal: {
          DEFAULT: "#0C6B5E",
          deep: "#08463E",
          tint: "#E1EFEB",
        },
        // Muted marigold gold, a data/₹ highlight accent (premium, not neon).
        gold: {
          DEFAULT: "#C0871B",
          tint: "#F6ECD5",
        },
      },
      fontFamily: {
        // Characterful display used with restraint …
        display: ['"Bricolage Grotesque"', "system-ui", "sans-serif"],
        // … reliable body workhorse …
        sans: ["Inter", "system-ui", "sans-serif"],
        // … and mono for data: node labels, ₹ amounts.
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: {
        card: "14px",
        control: "10px",
      },
      boxShadow: {
        card: "0 1px 2px rgba(20, 33, 29, 0.04), 0 6px 20px -8px rgba(20, 33, 29, 0.10)",
        raise: "0 8px 30px -10px rgba(12, 107, 94, 0.28)",
      },
    },
  },
  plugins: [],
} satisfies Config;
