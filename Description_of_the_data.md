# Description of the data files

*All data has been anonymized. You must nevertheless not share this data-set with any outsiders.*

* The information is contained in two datafiles:
    * [*solditems_encoded_stage2.csv* contains sales information](#information-on-product-sales)
    * [*content_encoded_stage2.csv* contains information on product features](#information-on-the-properties-of-the-products)
* Note that not all variables were originally specified for each column, but there are no NA (missing) values in the datafiles:
    * For categorical variables, missing values include a 0 in all one-hot encoded columns of the variable.
    * For numerical variables, missing values were simply filled with the column mean (the most simple imputation).

## Information on product sales

1) Datafile: **solditems_encoded_stage2.csv**
2) Each row corresponds to the sale of one specific product token.
3) Apart from the first two columns,
    * each *numerical variable* has been normalized to the [0,1] interval;
    * each *categorical variable* (including *binary variables*) has been one-hot encoded. 

Description of the columns:

* `product_sid`: The unique identifier of a product model. You can think of this as the type, and the rows of the sales data correspond to tokens of these types.
    * This identifier corresponds to the `ProductId` in the datafile describing properties of product models.
* `created_date_id`: This is the date in 'YYYY-MM-DD' format when the specific product token was sold.
* `sales_item_price_created`: The actual price at which the product token was sold.
* `sales_item_price`:  The price of the product token.
* `sales_voucher_created`: The actual price of a voucher for the product token.
* `sales_voucher`: The price of a voucher for the product token.
* `sales_value_created`: The actual price paid by the buyer.
* `sales_value`: The price to be payed by the buyer.
* `created_year`: The year of the sale.
* `created_month`: The month of the sale.
* `created_weekday`: The weekday of the sale *originally* encoded from 0 (Monday) to 6 (Sunday).
* `days_since_first_sold`: The number of days since the frst sale happened
* `days_since_release`: The number of days since the product was released.
* `Ratio_SalesItemPrice_per_created`: Ratio of the *sales_item_price / sales_item_price_created* variables.
* `Ratio_SalesValue_per_created`:  Ratio of the *sales_value / sales_value_created* variables.
* `Ratio_SalesItemPrice_per_SalesValue`: Ratio of the *sales_item_price / sales_value* variables.
* `Ratio_SalesItemPriceCreated_per_SalesValueCreated`: Ratio of the *sales_item_price_created / sales_value_created* variables.
* `channel_sid_N`: One-hot encoded column of the *channel_sid* variable which specifies the channel (e.g., web browser) on which the sale happened.
* `returned_date_id_N`:  One-hot encoded column of the *returned_date* variable which specifies if a product token was returned (N=1) or not (N=0).

Note that the ratio values were included so that the original relation between the variables in question is not lost after scaling the columns.

Note also that some of the columns have been pre-computed for your convenience only; if you like, you can compute them or other derived variables yourself.

## Information on the properties of the products

1) Datafile: **content_encoded_stage2.csv**
2) Each row corresponds to a product model (that is, a type whose tokens are rows of the sales datafile).
3) Apart from the first two columns,
    * each *numerical variable* has been normalized to the [0,1] interval;
    * each *categorical variable* (including *binary variables*) has been one-hot encoded. 

Description of the columns:

* `ProductId`: The unique identifier of a product model.
    * This identifier corresponds to the `product_sid` in the datafile describing product sales.
* `ProductNameCleaned`: the product name without the color specification (encoded here as an integer). 
This means that several product models (with different product ID's) may be assigned to the same ProductNameCleaned.  
* `ReleaseYear`, `ReleaseMonth`, `ReleaseDayOfWeek` encode the year, month and weekday (originally from 0=Monday to 6=Sunday) of the date when the product was released.
* `DisplaySize_in`, `Storage_GB`, `CPUSpeed_GHz`, `DisplayHeight_px`, `DisplayResolution_MP`, `DisplayWidth_px`, `FrontCamResolution_MP`, 
`MainDescription_NumOfWords`, `Network_NumOfFreqs`, `RAM_GB` , `RearCamResolution_MP` , 
`Size_Depth` , `Size_Height` , `Size_Width` , `Weight` , `Ratio_HeightPerWidth` , `Ratio_DepthPerWidth` 
are numerical variables describing different properties of a product. Most of these are self-explanatory except maybe for the following:
    * `MainDescription_NumOfWords` encoded the number of words in the textual description of the product.
    * `Network_NumOfFreqs` encoded the number of frequencies the product is capable of handling.
* The rest of the columns are one-hot-encoded columns of categorical variables capturing different properties of the products. Again, the names of the variables should be self-explanatory.
    
