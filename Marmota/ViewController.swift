import UIKit
import AVFoundation
import Vision
import SwiftyJSON

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    let label: UILabel = {
        let label = UILabel()
        label.textColor = .white
        label.translatesAutoresizingMaskIntoConstraints = false
        label.text = "Not Dog"
        label.font = label.font.withSize(20)
        label.numberOfLines = 3
        return label
    }()
    var buffer: CMSampleBuffer!
    var dogLabels: [String] = []
    
    func setupUILabel() {
        label.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        label.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -50).isActive = true
    }
    
    func setupDogLabels() {
        if let path = Bundle.main.path(forResource: "Labels", ofType: "json") {
            do {
                let data = try Data(contentsOf: URL(fileURLWithPath: path), options: .alwaysMapped)
                let jsonData = try JSON(data: data)
                for (_,subJson):(String, JSON) in jsonData {
                    let dogName = subJson["label"].stringValue
                    dogLabels.append(dogName)
                }
            } catch let error {
                print("parse error: \(error.localizedDescription)")
            }
        } else {
            print("Labels file not found")
        }
    }
    
    func setupCaptureSession() {
        let captureSession = AVCaptureSession()
        let availableDevices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: AVMediaType.video, position: .back).devices
        do {
            if let captureDevice = availableDevices.first {
                captureSession.addInput(try AVCaptureDeviceInput(device: captureDevice))
            }
        } catch {
            print(error.localizedDescription)
        }
        
        let captureOutput = AVCaptureVideoDataOutput()
        captureSession.addOutput(captureOutput)
        
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.frame
        view.layer.addSublayer(previewLayer)
        
        captureSession.startRunning()
        
        captureOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
    }
    
    private lazy var detectDogRequest: VNCoreMLRequest? = {
        guard let model = try? VNCoreMLModel(for: Resnet50().model) else { return nil }
        let request = VNCoreMLRequest(model: model) { (request, error) in
            guard let observations = request.results else { return }
            let results = observations
                .prefix(through: 4)
                .compactMap({ $0 as? VNClassificationObservation })
                .map({$0})
            
            var isDog = false
            for result in results {
                if self.dogLabels.contains(result.identifier) {
                    isDog = true
                }
            }
            
            if isDog {
                self.classifyBreed(buffer: self.buffer)
            } else {
                DispatchQueue.main.async(execute: {
                    self.label.text = "Not Dog"
                })
            }
        }
        return request
    }()
    
    private lazy var detectBreedRequest: VNCoreMLRequest? = {
        guard let model = try? VNCoreMLModel(for: StudentDogModel().model) else { return nil }
        
        let request = VNCoreMLRequest(model: model) { request, error in
            if let observations = request.results as? [VNClassificationObservation] {
                let top3 = observations
                    .prefix(through: 2)
                    .sorted(by: { a, b -> Bool in a.confidence > b.confidence })
                    .map { String(format: "%@ (%3.2f%%)", $0.identifier, $0.confidence * 100) }
                
                DispatchQueue.main.async(execute: {
                    self.label.text = top3.joined(separator: "\n")
                })
            }
        }
        return request
    }()
    
    func classifyBreed(buffer: CMSampleBuffer) {
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(buffer) else { return }
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([detectBreedRequest!])
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer: CVPixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        self.buffer = sampleBuffer
        try? VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:]).perform([detectDogRequest!])
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupCaptureSession()
        view.addSubview(label)
        setupUILabel()
        setupDogLabels()
    }
}

