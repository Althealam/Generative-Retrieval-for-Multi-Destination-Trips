# Generative Retrieval for Multi-Destination Trips via RQ-VAE

This project implements a generative retrieval approach for predicting multi-destination trips using Residual Quantized Variational Autoencoder (RQ-VAE), based on the Booking.com Multi-Destination Trips dataset.

## Overview

Multi-destination trips involve travelers visiting multiple cities in a single journey. This project aims to predict the next destination city for incomplete multi-destination trips using advanced machine learning techniques.

## Dataset

The project uses the **Booking.com Multi-Destination Trips Dataset** from the WSDM WebTour 2021 Challenge.

The training dataset consists of over a million of anonymized hotel reservations, based on real data, with the following features: 
* user_id: User ID
* check-in: reservation check-in date
* checkout: reservation check-out date
* affiliate_id: An anonymized ID of affiliate channels where the booker came from (e.g., direct, some third party referrals, paid search engine, etc.)
* device_class: desktop/mobile
* booker_country: Country from which the reservation was made (anonymized)
* hotel_country: Country of the hotel (anonymized)
* city_id: city id of the hotel's city (anonymized)
* utrip_id: Unique identification


### Dataset Statistics
- **Training Set**: 1,166,835 bookings
- **Test Set**: 378,667 bookings (with 70,662 trips to predict)
- **Features**: user_id, checkin, checkout, city_id, device_class, affiliate_id, booker_country, hotel_country, utrip_id
- **Target**: Predict the final destination city for incomplete trips

## References

### Dataset
- **Data Source**: [Booking.com Multi-Destination Trips Dataset](https://github.com/bookingcom/ml-dataset-mdt)
- **Paper**: [Multi-Destination Trip Dataset](https://dl.acm.org/doi/10.1145/3404835.3463240)
- **Challenge**: Booking.com WSDM WebTour 2021 Challenge
- **Conference**: [WSDM 2021](https://ceur-ws.org/Vol-2855/)

### Citation
If you use this dataset or code, please cite:
```bibtex
@inproceedings{ravagli2021booking,
  title={Booking.com WSDM WebTour 2021 Challenge},
  author={Ravagli, Niek and others},
  booktitle={WSDM WebTour Workshop},
  year={2021}
}
```


## License

This project is provided for academic and research purposes. Please refer to the original dataset license for data usage terms.

## Acknowledgments

- Booking.com for providing the multi-destination trips dataset
- WSDM WebTour 2021 Challenge organizers