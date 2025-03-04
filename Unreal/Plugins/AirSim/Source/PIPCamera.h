#pragma once

#include "CoreMinimal.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Camera/CameraActor.h"
#include "Materials/Material.h"
#include "Runtime/Core/Public/PixelFormat.h"
#include "common/ImageCaptureBase.hpp"
#include "common/common_utils/Utils.hpp"
#include "common/AirSimSettings.hpp"
#include "NedTransform.h"

#include "Materials/MaterialParameterCollection.h"
#include "Materials/MaterialParameterCollectionInstance.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "PIPCamera.generated.h"

UCLASS()
class AIRSIM_API APIPCamera : public ACameraActor
{
    GENERATED_BODY()

public:
    typedef msr::airlib::ImageCaptureBase::ImageType ImageType;
    typedef msr::airlib::AirSimSettings AirSimSettings;
    typedef AirSimSettings::CameraSetting CameraSetting;

    APIPCamera();

    virtual void PostInitializeComponents() override;
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;

    void showToScreen();
    void disableAll();
    void disableAllPIP();
    void disableMain();
    void onViewModeChanged(bool nodisplay);

    void setCameraTypeEnabled(ImageType type, bool enabled);
    bool getCameraTypeEnabled(ImageType type) const;
    void setupCameraFromSettings(const APIPCamera::CameraSetting& camera_setting, const NedTransform& ned_transform);
    void setCameraPose(const FTransform& pose);
    void setCameraFoV(float fov_degrees);
	void setCameraImageSize(const float width, const float height);
	void setCameraPostProcess(
		const float blend_weight,
		const float bloom_intensity,
		const int auto_exposure_method, const float auto_exposure_bias,
		const float auto_exposure_speed_up, const float auto_exposure_speed_down,
		const float auto_exposure_max_brightness, const float auto_exposure_min_brightness,
		const float lens_flare_intensity,
		const float ambient_occlusion_intensity, const float ambient_occlusion_radius,
		const FLinearColor& indirect_lighting_color, const float indirect_lighting_intensity,
		const float motion_blur_amount
	);

    msr::airlib::ProjectionMatrix getProjectionMatrix(const APIPCamera::ImageType image_type) const;

    USceneCaptureComponent2D* getCaptureComponent(const ImageType type, bool if_active);
    UTextureRenderTarget2D* getRenderTarget(const ImageType type, bool if_active);

    msr::airlib::Pose getPose() const;

    UPROPERTY()
    UMaterialParameterCollection* distortion_param_collection_;
    UPROPERTY()
    UMaterialParameterCollectionInstance* distortion_param_instance_;

private: //members
    UPROPERTY()
    TArray<USceneCaptureComponent2D*> captures_;
    UPROPERTY()
    TArray<UTextureRenderTarget2D*> render_targets_;

    UPROPERTY()
    UCameraComponent* camera_;
    //TMap<int, UMaterialInstanceDynamic*> noise_materials_;
    //below is needed because TMap doesn't work with UPROPERTY, but we do have -ve index
    UPROPERTY()
    TArray<UMaterialInstanceDynamic*> noise_materials_;
    UPROPERTY()
    TArray<UMaterialInstanceDynamic*> distortion_materials_;
    UPROPERTY()
    UMaterial* noise_material_static_;
    UPROPERTY()
    UMaterial* distortion_material_static_;

    std::vector<bool> camera_type_enabled_;
    FRotator gimbald_rotator_;
    float gimbal_stabilization_;
    const NedTransform* ned_transform_;
    TMap<int, EPixelFormat> image_type_to_pixel_format_map_;

private: //methods
    typedef common_utils::Utils Utils;
    typedef AirSimSettings::CaptureSetting CaptureSetting;
    typedef AirSimSettings::NoiseSetting NoiseSetting;

    static unsigned int imageTypeCount();
    void enableCaptureComponent(const ImageType type, bool is_enabled);
    static void updateCaptureComponentSetting(USceneCaptureComponent2D* capture, UTextureRenderTarget2D* render_target,
                                              bool auto_format, const EPixelFormat& pixel_format, const CaptureSetting& setting, const NedTransform& ned_transform,
                                              bool force_linear_gamma);
    void setNoiseMaterial(int image_type, UObject* outer, FPostProcessSettings& obj, const NoiseSetting& settings);
    void setDistortionMaterial(int image_type, UObject* outer, FPostProcessSettings& obj);
    static void updateCameraPostProcessingSetting(FPostProcessSettings& obj, const CaptureSetting& setting);
    static void updateCameraSetting(UCameraComponent* camera, const CaptureSetting& setting, const NedTransform& ned_transform);
};
