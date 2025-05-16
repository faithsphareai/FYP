import { View, Text ,Pressable ,StyleSheet} from 'react-native'
import React from 'react'

const SearchButton = ({onPress}) => {
  return (
    <View style={styles.outerCircle}>
          <View style={styles.secondOuterCircle}>
            <Pressable
            onPress={onPress} 
              style={styles.FindMosquesNearbyButton}
              android_ripple={{ 
                color: 'rgba(22, 65, 96, 0.5)', 
                borderless: false, // Set to false to make the ripple effect have bounds
                radius: 50, // Set the radius of the ripple effect (adjust as needed)
              }}
            > 
              <Text style={styles.btnText}>Search</Text>
            </Pressable>
          </View>
    </View>
  )
}

const styles = StyleSheet.create({
    outerCircle:{
        position:'absolute',
        padding:12,
        borderRadius:100,
        backgroundColor:'rgba(149, 187, 214,0.3)',
        top:10,
      },
      secondOuterCircle:{
        backgroundColor:'rgba(88, 147, 188,0.5)', 
        borderRadius:70,
        padding: 12,
      },
      FindMosquesNearbyButton:{
        borderRadius:50,
        height:95,
        width:95,
        backgroundColor:'#33658A',
        alignItems:'center',
        justifyContent:'center',
        elevation:10, 
      },
      btnText:{
        color:'white',
        fontSize:17,
        fontWeight:'bold',
      },
})

export default SearchButton