//async storage
import AsyncStorage from '@react-native-async-storage/async-storage'
//expo location
import * as Location from 'expo-location';


export default getCurrentLocation = async ()=>{
    let coords =  await AsyncStorage.getItem("location");
    if(coords==null){
      let {status} = await Location.requestForegroundPermissionsAsync();
      if(status!=="granted"){
        alert("please grant permission in order to get app functional");
        return;
      }
      let currentLocation = await Location.getCurrentPositionAsync({});
      await AsyncStorage.setItem("location",JSON.stringify(currentLocation));
      coords = currentLocation;
    }else{
      coords = JSON.parse(coords);
    }
    return coords; 
}