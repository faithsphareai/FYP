import { create } from 'zustand';

const useNearbymosquesstore = create((set, get) => ({
  mosquesData: [],
  setMosques: (mosques) => set({ mosquesData: mosques }),
  getMosqueById: (id) => {
    return get().mosquesData.find(mosque => mosque.id === id);
  }
}));

export default useNearbymosquesstore;
