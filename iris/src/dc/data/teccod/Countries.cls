/// 
Class dc.data.teccod.Countries Extends %Persistent [ ClassType = persistent, DdlAllowed, Final, Owner = {irisowner}, ProcedureBlock, SqlRowIdPrivate, SqlTableName = Countries ]
{

Property name As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 2 ];

Property alpha2 As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 3, SqlFieldName = alpha_2 ];

Property alpha3 As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 4, SqlFieldName = alpha_3 ];

Property countrycode As %Library.Integer(MAXVAL = 2147483647, MINVAL = -2147483648) [ SqlColumnNumber = 5, SqlFieldName = country_code ];

Property iso31662 As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 6, SqlFieldName = iso_3166_2 ];

Property region As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 7 ];

Property subregion As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 8, SqlFieldName = sub_region ];

Property intermediateregion As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 9, SqlFieldName = intermediate_region ];

Property regioncode As %Library.Integer(MAXVAL = 2147483647, MINVAL = -2147483648) [ SqlColumnNumber = 10, SqlFieldName = region_code ];

Property subregioncode As %Library.Integer(MAXVAL = 2147483647, MINVAL = -2147483648) [ SqlColumnNumber = 11, SqlFieldName = sub_region_code ];

Property intermediateregioncode As %Library.String(MAXLEN = 250) [ SqlColumnNumber = 12, SqlFieldName = intermediate_region_code ];

Parameter USEEXTENTSET = 1;

/// Bitmap Extent Index auto-generated by DDL CREATE TABLE statement.  Do not edit the SqlName of this index.
Index DDLBEIndex [ Extent, SqlName = "%%DDLBEIndex", Type = bitmap ];

Storage Default
{
<Data name="CountriesDefaultData">
<Value name="1">
<Value>name</Value>
</Value>
<Value name="2">
<Value>alpha2</Value>
</Value>
<Value name="3">
<Value>alpha3</Value>
</Value>
<Value name="4">
<Value>countrycode</Value>
</Value>
<Value name="5">
<Value>iso31662</Value>
</Value>
<Value name="6">
<Value>region</Value>
</Value>
<Value name="7">
<Value>subregion</Value>
</Value>
<Value name="8">
<Value>intermediateregion</Value>
</Value>
<Value name="9">
<Value>regioncode</Value>
</Value>
<Value name="10">
<Value>subregioncode</Value>
</Value>
<Value name="11">
<Value>intermediateregioncode</Value>
</Value>
</Data>
<DataLocation>^dc.data.teccod.Countries.1</DataLocation>
<DefaultData>CountriesDefaultData</DefaultData>
<ExtentLocation>^dc.data.teccod.Countries</ExtentLocation>
<ExtentSize>0</ExtentSize>
<IdFunction>sequence</IdFunction>
<IdLocation>^dc.data.teccod.Countries.1</IdLocation>
<Index name="DDLBEIndex">
<Location>^dc.data.teccod.Countries.2</Location>
</Index>
<Index name="IDKEY">
<Location>^dc.data.teccod.Countries.1</Location>
</Index>
<IndexLocation>^dc.data.teccod.Countries.I</IndexLocation>
<StreamLocation>^dc.data.teccod.Countries.S</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}