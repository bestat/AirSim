// Fill out your copyright notice in the Description page of Project Settings.


#include "MetahumanAnimInstance.h"

void UMetahumanAnimInstance::SetMetahumanPose(
	const FVector& LeftHand_IKPosition_, const FRotator& LeftHand_Rotation_, const FVector& RightHand_IKPosition_, const FRotator& RightHand_Rotation_
)
{
	this->LeftHand_IKPosition = LeftHand_IKPosition_;
	this->LeftHand_Rotation = LeftHand_Rotation_;
	this->RightHand_IKPosition = RightHand_IKPosition_;
	this->RightHand_Rotation = RightHand_Rotation_;
}