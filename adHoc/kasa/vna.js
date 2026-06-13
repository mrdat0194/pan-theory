<script>
window._prev_dxData=window._prev_dxData||window.dxData;
var _gtm_Vars = [];
var transactionDeduper = {
   keyName: '_gtm_transaction_ids',
   cookieExpiresDays: 30
};

var readFromStorage = function(key) {
  if (!window.Storage) {
    // From: https://stackoverflow.com/a/15724300/2367037
    var value = '; ' + document.cookie;
    var parts = value.split('; ' + key + '=');
    if (parts.length === 2) return parts.pop().split(';').shift();
  } else {
    return window.localStorage.getItem(key);
  }
};

var writeToStorage = function(key, value, expireDays) {
  if (!window.Storage) {
    var expiresDate = new Date();
    expiresDate.setDate(expiresDate.getDate() + expireDays);
    document.cookie = key + '=' + value + ';expires=' + expiresDate.toUTCString();
  } else {
    window.localStorage.setItem(key, value);
  }
};

if({{Custom Event}}=='cartUpdated') {
    if (typeof _prev_dxData['last_cart'] != 'undefined' && _prev_dxData['last_cart'] != null) {
        var new_cart = getProductList();
        var diff_cart = [];
        var last_cart = _prev_dxData['last_cart'];
        for(var i=0; i<last_cart.length; i++) {
            for(var j=0; j<new_cart.length; j++) {
                if(last_cart[i]['id']==new_cart[j]['id']) {
                    break;
                }
                if(j==new_cart.length-1) {
                    diff_cart.push(last_cart[i]);
                }
            }
        }
        if(diff_cart.length > 0) {
            var ecommerceObject = {
                'ecommerce': {
                    'remove': {
                        'products': diff_cart
                    }
                },
                'event' : 'productRemoveFromCart'
            };
            dataLayer.push(ecommerceObject);
        }
    }
    // check cartupdate voi aria-disabled ="false"
    _prev_dxData['last_cart'] = getProductList();
}
if({{Custom Event}}=='PageViewLoaded' && dxData.page.pageInfo.pageName=='passengers') {
    checkoutStep(1);
}
if({{Custom Event}}=='PageViewLoaded' && dxData.page.pageInfo.pageName=='ancillaries') {
    checkoutStep(2);
}
if({{Custom Event}}=='PageViewLoaded' && dxData.page.pageInfo.pageName=='seat-selection') {
    checkoutStep(3);
}
if({{Custom Event}}=='PageViewLoaded' && dxData.page.pageInfo.pageName=='payment') {
    checkoutStep(4);
}
if({{Custom Event}} == 'transactionCompleted') {
    _gtm_recieptPage();
}
// Custom add to cart
if({{Custom Event}}=='gtm.click' && dxData.page.pageInfo.pageName=='flight-selection') {
    var elementClickCart = {{Click Element}};
    var objectButtonCart = elementClickCart.closest('.brand-select-button');

    if (typeof objectButtonCart != 'undefined' && objectButtonCart != null) {
        var wrapDetail = objectButtonCart.closest('.spark-panel__content');

        if (typeof wrapDetail != 'undefined' && wrapDetail != null) {

            var time_airport = wrapDetail.getElementsByClassName('time-airport');
            var origin = time_airport[0].getElementsByClassName('airport')[0].innerHTML;
            var destination = time_airport[1].getElementsByClassName('airport')[0].innerHTML;
            var wrapBrand = wrapDetail.getElementsByClassName('itinerary-part-summary-details')[0].getElementsByClassName('dxp-image')[0];


            var object_html_header = wrapDetail.getElementsByClassName('itinerary-part-offer-header');
            var stop_airports = object_html_header[0].getElementsByClassName('stop-airports');

            if (stop_airports.length > 0) {
                var stop_airports_text = stop_airports[0].innerHTML;
                stop_airports_text = stop_airports_text.replace('(', '');
                stop_airports_text = stop_airports_text.replace(')', '');
                var stop_airports_arr = stop_airports_text.split(',');


                var products = [];
                var next_start = origin;
                var next_stop = destination;
                for (var i = 0; i <= stop_airports_arr.length; i++) {
                    if(i == stop_airports_arr.length){
                        next_stop = destination;
                    } else {
                        next_stop = stop_airports_arr[i];
                    }

                    products.push({
                        'id': wrapDetail.getElementsByClassName('flight-number')[i].innerHTML, // VN 223
                        'name': 'Flight::' + next_start + ':' + next_stop,
                        'category': 'Flight',
                        'quantity': dxData.product[0].productInfo.passengerCount,
                        'brand': wrapBrand.getAttribute('alt')
                    });

                    next_start = stop_airports_arr[i];
                }
            } else {
                var products = [
                    {
                        'id': wrapDetail.getElementsByClassName('flight-number')[0].innerHTML, // VN 223
                        'name': 'Flight::' + origin + ':' + destination,
                        'category': 'Flight',
                        'quantity': dxData.product[0].productInfo.passengerCount,
                        'brand': wrapBrand.getAttribute('alt')
                    }
                ];
            }


            var ecommerceObject = {
                'ecommerce': {
                    'add': {
                        'products': products
                    }
                },
                'event' : 'productAddToCart'
            };
            dataLayer.push(ecommerceObject);
        }
    }// Custom add to cart
}

function checkoutStep(step) {
    var ecommerceObject = {
        'event': 'productCheckout',
        'ecommerce': {
            'checkout': {
                'products': getProductList(),
                'actionField': {
                    'step': step
                }
            }
        }
    };
    dataLayer.push(ecommerceObject);
}

function getProductList() {
    var productList = [];
    var IBE_DX_Baggage_Price = 0;
    var IBE_DX_Seat_Price = 0;
    var IBE_DX_Price = 0;
    try {
        if(typeof dxData.transaction != 'undefined' && typeof dxData.transaction.pnrInfo.itinerary.itineraryParts != 'undefined') {
          for(var i=0; i<dxData.transaction.cart.products.length; i++) {
            //Flight
            if(dxData.transaction.cart.products[i].label == "farePrice") {
              for(var j=0; j<dxData.transaction.cart.products[i].breakdownElements.length; j++) {
                var element = dxData.transaction.cart.products[i].breakdownElements[j];
                for(var k=0; k < element.breakdownElementAssignment.travelPart.segments.length; k++) {
                  var segment = element.breakdownElementAssignment.travelPart.segments[k];
                  productList.push({
                    'id': segment.flight.operatingAirlineCode + ' ' + segment.flight.flightNumber,
                    'name': 'Flight::' + segment.origin + ':' + segment.destination,
                    'price': element.price.amount,
                    'quantity': 1,
                    'brand': segment.flight.operatingAirlineCode,
                    'category': 'Flight',
                    'variant': segment.bookingClass
                  });
                  IBE_DX_Price += parseFloat(element.price.amount);
                }
              }
            }
          }
          //Ancillary
          if(typeof dxData.transaction.transactionProducts != 'undefined') {
              for(var i=0; i<dxData.transaction.transactionProducts.length; i++) {
                var element = dxData.transaction.transactionProducts[i];
                if(element.category=="Ancillary" && element.name == "Seat") {
                    productList.push({
                        'id': 'Seat',
                        'name': 'Seat',
                        'price': element.price,
                        'quantity': element.quantity,
                        'category': 'Seat',
                        'variant': element.variant
                      });
                    IBE_DX_Seat_Price += parseFloat(element.price)*parseFloat(element.quantity);
                    } else if(element.category=="Ancillary" && element.name == "insurance") {
                    productList.push({
                        'id': 'Ancillary::Insurance',
                        'name': 'Ancillary::Insurance',
                        'price': element.price,
                        'quantity': element.quantity,
                        'category': 'Insurance',
                        'variant': element.variant
                      });
                } else if(element.category!="Flight") {
                    productList.push({
                        'id': 'Ancillary::'+element.name,
                        'name': 'Ancillary::'+element.name,
                        'price': element.price,
                        'quantity': element.quantity,
                        'category': element.name,
                        'variant': element.variant
                      });
                    IBE_DX_Baggage_Price += parseFloat(element.price)*parseFloat(element.quantity);
                }
              }
            } else if(typeof dxData.transaction.pnrInfo.travelPartsAdditionalDetails[0].passengers[0].ancillaries[0].ancillaryCode != 'undefined') {
                var ancillaryCodes = [];
                for(var i=0; i < dxData.transaction.pnrInfo.travelPartsAdditionalDetails.length; i++) {
                    var passengers = dxData.transaction.pnrInfo.travelPartsAdditionalDetails[i].passengers;
                    for(var j=0; j < passengers.length; j++) {
                      if(typeof passengers[j]["checkedInBaggage"] != 'undefined') {
                        ancillaryCodes[passengers[j].ancillaries[0].ancillaryCode] = {'ancillaryCode': passengers[j].ancillaries[0].ancillaryCode, 'assignedQuantity': passengers[j].ancillaries[0].assignedQuantity, 'baggageAllowanceType': passengers[j].checkedInBaggage.baggageAllowanceDefinition["0"].baggageAllowanceType};
                                }
                      if(typeof passengers[j]["seat"] != 'undefined') {
                        ancillaryCodes[passengers[j].ancillaries[0].ancillaryCode] = {'ancillaryCode': passengers[j].ancillaries[0].ancillaryCode, 'assignedQuantity': passengers[j].ancillaries[0].assignedQuantity, 'seatCode': passengers[j].seat.seatCode};
                                }
                    }
                }
                //Baggage
                for(var i=0; i<dxData.transaction.cart.products.length; i++) {
                    if(dxData.transaction.cart.products[i].label == "ancillariesPrice") {
                        for(var j=0; j<dxData.transaction.cart.products[i].breakdownElements.length; j++) {
                           var element = dxData.transaction.cart.products[i].breakdownElements[j];
                           if(ancillaryCodes.hasOwnProperty(element.label)) {
                                productList.push({
                                  'id': 'Ancillary::' + element.label,
                                  'name': 'Ancillary::' + element.label,
                                  'price': element.price.amount,
                                  'quantity': ancillaryCodes[element.label].assignedQuantity,
                                  'category': 'Ancillary',
                                  'brand': ancillaryCodes[element.label].baggageAllowanceType
                                });
                                IBE_DX_Baggage_Price += parseFloat(element.price.amount)*parseFloat(ancillaryCodes[element.label].assignedQuantity);
                            }
                        }
                    }
                    if(dxData.transaction.cart.products[i].label == "seatsPrice") {
                        for(var j=0; j<dxData.transaction.cart.products[i].breakdownElements.length; j++) {
                            var element = dxData.transaction.cart.products[i].breakdownElements[j];
                            if(ancillaryCodes.hasOwnProperty(element.label)) {
                                productList.push({
                                  'id': 'Seat',
                                  'name': 'Seat',
                                  'price': element.price.amount,
                                  'quantity': ancillaryCodes[element.label].assignedQuantity,
                                  'category': 'Seat',
                                  'variant': element.breakdownElementAssignment.travelPart.origin + '-' + element.breakdownElementAssignment.travelPart.destination + ':' + ancillaryCodes[element.label].seatCode
                                });
                            IBE_DX_Seat_Price += parseFloat(element.price.amount)*parseFloat(ancillaryCodes[element.label].assignedQuantity);
                            }
                        }
                    }
                }
            }
        } else if(typeof dxData.cart != 'undefined') {
          //Flight
          if(typeof dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts != 'undefined' && dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts.length > 0) {
            for(var i=0; i < dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts.length; i++) {
              for(var j=0; j < dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments.length; j++) {
                productList.push({
                  'id': dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].flight.operatingAirlineCode + ' ' + dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].flight.flightNumber,
                  'name': 'Flight::' + dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].origin + ':' + dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].destination,
                  'price': 0,
                  'quantity': dxData.product[0].productInfo.passengerCount,
                  'brand': dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].flight.operatingAirlineCode,
                  'category': 'Flight',
                  'variant': dxData.cart.items[0].breakdownElementAssignment.travelPart.itineraryParts[i].segments[j].bookingClass
                });
              }
            }
          }
          //Extra Baggage
          if(typeof dxData.cart.items[3].breakdownElements != 'undefined' && dxData.cart.items[3].breakdownElements.length > 0) {
            for(var i=0; i<dxData.cart.items[3].breakdownElements.length; i++) {
              productList.push({
                'id': 'Ancillary::' + dxData.cart.items[3].breakdownElements[i].name,
                'name': 'Ancillary::' + dxData.cart.items[3].breakdownElements[i].name,
                'price': dxData.cart.items[3].breakdownElements[i].price[0][0].amount,
                'quantity': dxData.cart.items[3].breakdownElements[i].quantity,
                'brand': dxData.cart.items[3].breakdownElements[i].ancillaryType,
                'category': 'Ancillary',
                'variant': dxData.cart.items[3].breakdownElements[i].passengerLabel
              });
              IBE_DX_Baggage_Price += parseFloat(dxData.cart.items[3].breakdownElements[i].price[0][0].amount)*parseFloat(dxData.cart.items[3].breakdownElements[i].quantity);
            }
          }
          //Seat
          if(typeof dxData.cart.items[5].breakdownElements != 'undefined' && dxData.cart.items[5].breakdownElements.length > 0) {
            for(var i=0; i<dxData.cart.items[5].breakdownElements.length; i++) {
              var seat_price = (dxData.cart.items[5].breakdownElements[i].price.length > 0)?dxData.cart.items[5].breakdownElements[i].price[0][0].amount:0;
              productList.push({
                'id': 'Seat',
                'name': 'Seat',
                'price': seat_price,
                'quantity': 1,
                'category': 'Seat',
                'variant': dxData.cart.items[5].breakdownElements[i].origin + '-' + dxData.cart.items[5].breakdownElements[i].destination + ':' + dxData.cart.items[5].breakdownElements[i].seatNumber
                });
              IBE_DX_Seat_Price += parseFloat(seat_price);
            }
          }
        }
    } catch(err) {
        console.log("Get product list error: " + err.message);
    }
    _gtm_Vars['IBE-DX-Baggage-Price'] = IBE_DX_Baggage_Price;
    _gtm_Vars['IBE-DX-Seat-Price'] = IBE_DX_Seat_Price;
    _gtm_Vars['IBE-DX-Price'] = IBE_DX_Price;
    return productList;
}

function _gtm_recieptPage() {
    //Prevent duplicate transaction
    var transactionId = dxData.transaction.transactionId;
  var storedIds = JSON.parse(readFromStorage(transactionDeduper.keyName) || '[]');
    if (storedIds.indexOf(transactionId) > -1) {
      var d = new Date();
      var n = d.getTime();
      console.log("Prevent Duplicate Transactions " + n);
      return;
    }//Prevent duplicate transaction
    var revenue = dxData.transaction.transactionTotal;
    var revenue_by_miles = 0;
    var tax_by_miles = 0;
    var tax = 0;
    for(var i=0; i<dxData.transaction.cart.products.length; i++) {
        var p = dxData.transaction.cart.products[i];
        if(p.label == "taxesPrice") {
            tax = p.total[0][0].amount;
        }
    }
    var paymentCode = dxData.transaction.paymentOptionsDetails[0].paymentCode;
    var paymentType = dxData.transaction.paymentOptionsDetails[0].paymentType;
    var coupon = (typeof dxData.product[0].productInfo.promoCode != 'undefined')?dxData.product[0].productInfo.promoCode:"";
    var payment_key = 'PaymentM-' + dxData.product[0].productInfo.origin + '-' + dxData.product[0].productInfo.destination + '-' + dxData.product[0].productInfo.date + '-' + dxData.product[0].productInfo.date1;
    var payment_method =  readFromStorage(payment_key);
    var payment_detail = "";
    if(payment_method == 'SMARTLINK') {
        payment_key = 'PaymentD-' + dxData.product[0].productInfo.origin + '-' + dxData.product[0].productInfo.destination + '-' + dxData.product[0].productInfo.date + '-' + dxData.product[0].productInfo.date1;
        payment_detail = readFromStorage(payment_key);
    }
    if(dxData.product[0].productInfo.awardBooking==true) {
        for(var i=0; i<dxData.transaction.totalPaid; i++) {
            if(dxData.transaction.totalPaid[i]["currency"]=="FFCURRENCY") {
                revenue_by_miles = dxData.transaction.totalPaid[i]["amount"];
            }
            if(dxData.transaction.totalPaid[i]["currency"]==dxData.page.pageInfo.currency) {
                revenue = dxData.transaction.totalPaid[i]["amount"];
            }
        }
        for(var i=0; i<dxData.cart.items[6].total[0]; i++) {
            if(dxData.cart.items[6].total[0][i]["currency"]=="FFCURRENCY") {
                tax_by_miles = dxData.cart.items[6].total[0][i]["amount"];
            }
            if(dxData.cart.items[6].total[0][i]["currency"]==dxData.page.pageInfo.currency) {
                tax = dxData.cart.items[6].total[0][i]["amount"];
            }
        }
        dataLayer.push({
            'IBE-DX-Revenue-By-Miles': revenue_by_miles,
            'IBE-DX-Tax-By-Miles': tax_by_miles
        });
    }
    var actionFieldObject = {
        'id': dxData.transaction.transactionId,
        'affiliation': dxData.transaction.transactionAffiliation,
        'revenue': revenue,
        'tax': tax,
        'shipping': 0,
        'coupon': coupon
    };
    _gtm_Vars['IBE-DX-Baggage-Price'] = 0;
    _gtm_Vars['IBE-DX-Seat-Price'] = 0;
    var productList = getProductList();
    dataLayer.push({
        'IBE-DX-Revenue': revenue,
        'IBE-DX-Tax': tax,
        'IBE-DX-Payment-Code': paymentCode,
        'IBE-DX-Payment-Type': paymentType,
        'IBE-DX-Baggage-Price': _gtm_Vars['IBE-DX-Baggage-Price'],
        'IBE-DX-Seat-Price': _gtm_Vars['IBE-DX-Seat-Price'],
        'IBE-DX-Price': _gtm_Vars['IBE-DX-Price'],
        'IBE-DX-Payment-Method': payment_method,
        'IBE-DX-Payment-Detail': payment_detail
    });
    if (productList.length === 0) {
        //return;
    }
    //if (dxData.transaction.paymentOptionsDetails[0].paymentType != 'undefined' && dxData.transaction.paymentOptionsDetails[0].paymentType != null) {
    var ecommerceObject = {
        'event': 'transactionSuccess',
        'ecommerce': {
            'purchase': {
                'products': productList,
                'actionField': actionFieldObject
            }
        }
    }
    //};
    var taxes = {};
    var trans_products = dxData.transaction.cart.products;
    for (var i = 0; i < trans_products.length; i++) {
        if (trans_products[i]['label'] == 'taxesPrice') {
            taxes = {
                'IBE-DX-YR': 0,
                'IBE-DX-YQ': 0,
            };
            for (var jx = 0; jx < trans_products[i].breakdownElements.length; jx++) {
                switch (trans_products[i].breakdownElements[jx].label) {
                    case "YQ":
                        taxes["IBE-DX-YQ"] = trans_products[i].breakdownElements[jx].price.amount;
                        break;
                    case "YR":
                        taxes["IBE-DX-YR"] = trans_products[i].breakdownElements[jx].price.amount;
                        break;
                }
            }
            dataLayer.push(taxes);
            break;
        }
    }

    dataLayer.push(ecommerceObject);
    //Prevent duplicate transaction
    storedIds.push(transactionId);
    writeToStorage(transactionDeduper.keyName, JSON.stringify(storedIds), transactionDeduper.cookieExpiresDays);
}
window._prev_dxData = dxData;
</script>