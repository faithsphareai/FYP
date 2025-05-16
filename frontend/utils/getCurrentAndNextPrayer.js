export const getCurrentAndNextPrayer = (times) => {
    const fieldsToRemove = ["Firstthird", "Imsak", "Lastthird", "Midnight", "Sunrise", "Sunset"];
    times = Object.fromEntries(
        Object.entries(times).filter(([key]) => !fieldsToRemove.includes(key))
      );
    const now = new Date();
    const currentTime = now.getHours() * 60 + now.getMinutes();
    
    const prayerList = Object.entries(times)
        .map(([name, time]) => {
            const [hours, minutes] = time.split(':').map(Number);
            const totalMinutes = hours * 60 + minutes;
            return { name, time, totalMinutes };
        })
        .sort((a, b) => a.totalMinutes - b.totalMinutes);
    
    let currentPrayer = null;
    let nextPrayer = null;

    for (let i = 0; i < prayerList.length; i++) {
        if (currentTime < prayerList[i].totalMinutes) {
            nextPrayer = prayerList[i];
            currentPrayer = i === 0 ? prayerList[prayerList.length - 1] : prayerList[i - 1];
            break;
        }
    }

    if (!nextPrayer) {
        currentPrayer = prayerList[prayerList.length - 1];
        nextPrayer = prayerList[0];
    }

    return {
        currentPrayer: { name: currentPrayer.name, time: currentPrayer.time },
        nextPrayer: { name: nextPrayer.name, time: nextPrayer.time }
    };
};