import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function AsrWomen() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  // Keep existing asrSteps data
  const asrSteps = [
    {
      title: "Preparation",
      description: "Prepare for Asr prayer with proper covering and intention",
      details: [
        "Ensure proper covering - loose clothes covering entire body except face and hands",
        "Make sure prayer area is clean",
        "Face the Qibla direction",
        "Make wudu if not already done",
        "Keep body movements gentle and controlled",
        "Ensure prayer time has begun"
      ],
      image: images.niyyah
    },
    {
      title: "First Rakat",
      description: "Begin the first Farz rakat of Asr",
      details: [
        "Make intention for 4 Farz rakats of Asr",
        "Raise hands below shoulders for Takbir",
        "Place right hand over left below chest",
        "Keep feet close together",
        "Recite Thana quietly",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah"
      ],
      image: images.qiyam
    },
    {
      title: "Ruku and Sujood",
      description: "Complete the movements of first rakat",
      details: [
        "Bend for Ruku with back straight but not too low",
        "Keep arms close to sides in Ruku",
        "Say 'Subhana Rabbiyal Azeem' 3 times",
        "Rise saying 'Sami Allahu Liman Hamidah'",
        "For Sujood, go down keeping body compact",
        "Keep arms close to body in Sujood",
        "Say 'Subhana Rabbiyal Ala' 3 times in each Sujood"
      ],
      image: images.ruku
    },
    {
      title: "Second Rakat",
      description: "Complete the second rakat",
      details: [
        "Stand gracefully for second Rakat",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Complete Ruku and Sujood as before",
        "Sit for Qaida-e-Oola (first sitting)",
        "Recite At-Tahiyyat",
        "Rise carefully for third Rakat"
      ],
      image: images.sajda
    },
    {
      title: "Third and Fourth Rakat",
      description: "Complete the remaining rakats",
      details: [
        "In third and fourth rakats:",
        "Recite only Surah Al-Fatiha",
        "Complete Ruku and Sujood maintaining modest posture",
        "Keep all movements gentle and controlled",
        "Maintain proper covering throughout",
        "Keep limbs close to body in all positions"
      ],
      image: images.qiyam
    },
    {
      title: "Final Sitting and Salam",
      description: "Complete the prayer with final sitting",
      details: [
        "Sit in final Qaida keeping feet to right side",
        "Recite At-Tahiyyat",
        "Recite Durood Ibrahim",
        "Recite the closing duas",
        "Turn face to right saying 'Assalam-o-Alaikum'",
        "Turn face to left saying 'Assalam-o-Alaikum'",
        "Make any personal duas"
      ],
      image: images.tashahhud
    }
  ];

  // Keep existing handleNext and handlePrevious functions
  const handleNext = () => {
    if (currentStep < asrSteps.length - 1) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <View style={prayerStyles.container}>
      <View style={prayerStyles.navigation}>
        <TouchableOpacity 
          onPress={handlePrevious}
          disabled={currentStep === 0}
          style={[prayerStyles.navButton, currentStep === 0 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-back" size={24} color="#16a34a" />
        </TouchableOpacity>
        
        <Text style={prayerStyles.stepText}>
          Step {currentStep + 1} of {asrSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === asrSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === asrSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {asrSteps[currentStep].title}
          </Text>
          
          <Image 
            source={asrSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {asrSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {asrSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {asrSteps.map((_, index) => (
          <View
            key={index}
            style={[
              prayerStyles.progressDot,
              index === currentStep && prayerStyles.activeDot
            ]}
          />
        ))}
      </View>
    </View>
  );
}