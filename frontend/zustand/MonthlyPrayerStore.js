// stores/usePrayerSettingsStore.js
import { create } from 'zustand'

const usePrayerSettingsStore = create((set) => ({
  schoolShift: 'hanfi',                  // 'hanfi' or 'shafi'
  setSchoolShift: (shift) => set({ schoolShift: shift }),
}))

export default usePrayerSettingsStore
