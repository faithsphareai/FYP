import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function IshaWomen() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);
  const windowWidth = Dimensions.get('window').width;

  const ishaSteps = [
    {
      title: "Preparation",
      description: "Prepare for Isha prayer with proper covering",
      details: [
        "Ensure proper covering - loose clothes covering entire body except face and hands",
        "Make sure prayer area is clean",
        "Face the Qibla direction",
        "Make wudu if not already done",
        "Keep movements gentle and controlled",
        "Ensure prayer time has begun"
      ],
      image: images.niyyah
    },
    {
      title: "Initial Sunnah",
      description: "Begin with 4 Sunnah rakats",
      details: [
        "Make Niyyah for Sunnah prayer",
        "Keep hands below shoulders for Takbir",
        "Place hands on chest, right over left",
        "Keep feet together and body compact",
        "Recite Thana quietly",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha and another Surah",
        "Complete positions modestly"
      ],
      image: images.qiyam
    },
    {
      title: "Farz Prayer",
      description: "Perform 4 Farz rakats",
      details: [
        "Make new Niyyah for Farz",
        "Maintain modest positioning",
        "Keep movements controlled",
        "Recite required Surahs",
        "In Ruku, bend only as needed",
        "Keep arms close to body in Sujood",
        "Complete all positions with composure"
      ],
      image: images.ruku
    },
    {
      title: "Additional Sunnah",
      description: "Complete 2 Sunnah rakats",
      details: [
        "Make Niyyah for 2 Sunnah",
        "Keep movements gentle",
        "Maintain proper covering",
        "Complete positions modestly",
        "Recite required Surahs",
        "End with Tashahhud and Salam",
        "Keep composure throughout"
      ],
      image: images.sajda
    },
    {
      title: "Witr Prayer",
      description: "Perform 3 Witr rakats",
      details: [
        "Make Niyyah for Witr",
        "Keep movements controlled",
        "In third rakat after Surah:",
        "Raise hands gently for Qunoot",
        "Recite Dua Qunoot quietly",
        "Complete remaining positions",
        "End with Tashahhud and Salam"
      ],
      image: images.tashahhud
    }
  ];

  const handleNext = () => {
    if (currentStep < ishaSteps.length - 1) {
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
          Step {currentStep + 1} of {ishaSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === ishaSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === ishaSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {ishaSteps[currentStep].title}
          </Text>
          
          <Image 
            source={ishaSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {ishaSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {ishaSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {ishaSteps.map((_, index) => (
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