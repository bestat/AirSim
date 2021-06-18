// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Animation/AnimInstance.h"
#include "MetahumanAnimInstance.generated.h"

/**
 * 
 */
UCLASS()
class AIRSIM_API UMetahumanAnimInstance : public UAnimInstance
{
	GENERATED_BODY()

public:

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Bestat Custom")
	FVector LeftHand_IKPosition;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Bestat Custom")
	FRotator LeftHand_Rotation;
	
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Bestat Custom")
	FVector RightHand_IKPosition;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Bestat Custom")
	FRotator RightHand_Rotation;

public:
	void SetMetahumanPose(const FVector& LeftHand_IKPosition_, const FRotator& LeftHand_Rotation_, const FVector& RightHand_IKPosition_, const FRotator& RightHand_Rotation_);
};
