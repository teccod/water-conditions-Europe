/// 
Class dc.data.teccod.waterPollution Extends %Persistent [ DependsOn = (dc.data.teccod.waterBodyCategories, dc.data.teccod.waterObservedProperty, dc.data.teccod.waterResultUom, dc.data.teccod.waterValueNorm) ]
{

Property parameterWaterBodyCategory As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 2 ];

Property observedPropertyDeterminandCode As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 3 ];

Property procedureAnalysedFraction As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 4 ];

Property procedureAnalysedMedia As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 5 ];

Property resultUom As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 6 ];

Property phenomenonTimeReferenceYear As %Library.Integer(MAXVAL = 2147483647, MINVAL = -2147483648) [ SqlColumnNumber = 7 ];

Property parameterSamplingPeriod As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 8 ];

Property resultMeanValue As %Library.Double [ SqlColumnNumber = 9 ];

Property waterBodyIdentifier As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 10 ];

Property Country As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 11 ];

Property PopulationDensity As %Library.Double [ SqlColumnNumber = 12 ];

Property TerraMarineProtected20162018 As %Library.Double [ SqlColumnNumber = 13, SqlFieldName = TerraMarineProtected_2016_2018 ];

Property TouristMean19902020 As %Library.Double [ SqlColumnNumber = 14, SqlFieldName = TouristMean_1990_2020 ];

Property VenueCount As %Library.Double [ SqlColumnNumber = 15 ];

Property netMigration20112018 As %Library.Double [ SqlColumnNumber = 16, SqlFieldName = netMigration_2011_2018 ];

Property droughtsfloodstemperature As %Library.Double [ SqlColumnNumber = 17, SqlFieldName = droughts_floods_temperature ];

Property literacyRate20102018 As %Library.Double [ SqlColumnNumber = 18, SqlFieldName = literacyRate_2010_2018 ];

Property combustibleRenewables20092014 As %Library.Double [ SqlColumnNumber = 19, SqlFieldName = combustibleRenewables_2009_2014 ];

Property gdp As %Library.Double [ SqlColumnNumber = 20 ];

Property compositionfoodorganicwastepercent As %Library.Double [ SqlColumnNumber = 21, SqlFieldName = composition_food_organic_waste_percent ];

Property compositionglasspercent As %Library.Double [ SqlColumnNumber = 22, SqlFieldName = composition_glass_percent ];

Property compositionmetalpercent As %Library.Double [ SqlColumnNumber = 23, SqlFieldName = composition_metal_percent ];

Property compositionotherpercent As %Library.Double [ SqlColumnNumber = 24, SqlFieldName = composition_other_percent ];

Property compositionpapercardboardpercent As %Library.Double [ SqlColumnNumber = 25, SqlFieldName = composition_paper_cardboard_percent ];

Property compositionplasticpercent As %Library.Double [ SqlColumnNumber = 26, SqlFieldName = composition_plastic_percent ];

Property compositionrubberleatherpercent As %Library.Double [ SqlColumnNumber = 27, SqlFieldName = composition_rubber_leather_percent ];

Property compositionwoodpercent As %Library.Double [ SqlColumnNumber = 28, SqlFieldName = composition_wood_percent ];

Property compositionyardgardengreenwastepercent As %Library.Double [ SqlColumnNumber = 29, SqlFieldName = composition_yard_garden_green_waste_percent ];

Property wastetreatmentrecyclingpercent As %Library.Double [ SqlColumnNumber = 30, SqlFieldName = waste_treatment_recycling_percent ];

Property idwaterbodycategories As dc.data.teccod.waterBodyCategories [ SqlColumnNumber = 31, SqlFieldName = id_waterbodycategories ];

Property idwatername As dc.data.teccod.waterName [ SqlColumnNumber = 32, SqlFieldName = id_watername ];

Property idwaterobservedproperty As dc.data.teccod.waterObservedProperty [ SqlColumnNumber = 33, SqlFieldName = id_waterobservedproperty ];

Property idwaterresultuom As dc.data.teccod.waterResultUom [ SqlColumnNumber = 34, SqlFieldName = id_waterresultuom ];

Storage Default
{
<Data name="waterPollutionDefaultData">
<Value name="1">
<Value>parameterWaterBodyCategory</Value>
</Value>
<Value name="2">
<Value>observedPropertyDeterminandCode</Value>
</Value>
<Value name="3">
<Value>procedureAnalysedFraction</Value>
</Value>
<Value name="4">
<Value>procedureAnalysedMedia</Value>
</Value>
<Value name="5">
<Value>resultUom</Value>
</Value>
<Value name="6">
<Value>phenomenonTimeReferenceYear</Value>
</Value>
<Value name="7">
<Value>parameterSamplingPeriod</Value>
</Value>
<Value name="8">
<Value>resultMeanValue</Value>
</Value>
<Value name="9">
<Value>waterBodyIdentifier</Value>
</Value>
<Value name="10">
<Value>Country</Value>
</Value>
<Value name="11">
<Value>PopulationDensity</Value>
</Value>
<Value name="12">
<Value>TerraMarineProtected20162018</Value>
</Value>
<Value name="13">
<Value>TouristMean19902020</Value>
</Value>
<Value name="14">
<Value>VenueCount</Value>
</Value>
<Value name="15">
<Value>netMigration20112018</Value>
</Value>
<Value name="16">
<Value>droughtsfloodstemperature</Value>
</Value>
<Value name="17">
<Value>literacyRate20102018</Value>
</Value>
<Value name="18">
<Value>combustibleRenewables20092014</Value>
</Value>
<Value name="19">
<Value>gdp</Value>
</Value>
<Value name="20">
<Value>compositionfoodorganicwastepercent</Value>
</Value>
<Value name="21">
<Value>compositionglasspercent</Value>
</Value>
<Value name="22">
<Value>compositionmetalpercent</Value>
</Value>
<Value name="23">
<Value>compositionotherpercent</Value>
</Value>
<Value name="24">
<Value>compositionpapercardboardpercent</Value>
</Value>
<Value name="25">
<Value>compositionplasticpercent</Value>
</Value>
<Value name="26">
<Value>compositionrubberleatherpercent</Value>
</Value>
<Value name="27">
<Value>compositionwoodpercent</Value>
</Value>
<Value name="28">
<Value>compositionyardgardengreenwastepercent</Value>
</Value>
<Value name="29">
<Value>wastetreatmentrecyclingpercent</Value>
</Value>
<Value name="30">
<Value>idwaterbodycategories</Value>
</Value>
<Value name="31">
<Value>idwatername</Value>
</Value>
<Value name="32">
<Value>idwaterobservedproperty</Value>
</Value>
<Value name="33">
<Value>idwaterresultuom</Value>
</Value>
<Value name="34">
<Value>%%CLASSNAME</Value>
</Value>
</Data>
<DataLocation>^dc.data.teccod.wPoll.1</DataLocation>
<DefaultData>waterPollutionDefaultData</DefaultData>
<ExtentLocation>^dc.data.teccod.wPoll</ExtentLocation>
<ExtentSize>0</ExtentSize>
<IdFunction>sequence</IdFunction>
<IdLocation>^dc.data.teccod.wPoll.1</IdLocation>
<Index name="DDLBEIndex">
<Location>^dc.data.teccod.wPoll.2</Location>
</Index>
<Index name="IDKEY">
<Location>^dc.data.teccod.wPoll.1</Location>
</Index>
<IndexLocation>^dc.data.teccod.wPoll.I</IndexLocation>
<StreamLocation>^dc.data.teccod.wPoll.S</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
