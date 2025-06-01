import React from 'react';
import { SunIcon, MoonIcon } from '@heroicons/react/24/solid';

function ThemeToggle({ toggleTheme, theme }) {
  return (
    <button
      onClick={toggleTheme}
      className="fixed top-4 right-4 p-2 bg-primary text-white rounded-full"
    >
      {theme === 'light' ? <MoonIcon className="h-6 w-6" /> : <SunIcon className="h-6 w-6" />}
    </button>
  );
}

export default ThemeToggle;