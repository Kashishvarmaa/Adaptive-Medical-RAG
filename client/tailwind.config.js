/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#4285f4',
        secondary: '#34a853',
        accent: '#fbbc04',
        danger: '#b81c0f',
      },
    },
  },
  plugins: [],
};
