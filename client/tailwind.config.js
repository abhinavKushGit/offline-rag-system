/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["'Syne'", "sans-serif"],
        mono:    ["'DM Mono'", "monospace"],
        body:    ["'Literata'", "Georgia", "serif"],
      },
    },
  },
  plugins: [],
};