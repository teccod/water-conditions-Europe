/// 
Class dc.teccod.cube.waterPollutionCube Extends %DeepSee.CubeDefinition [ DependsOn = dc.data.teccod.waterPollution, ProcedureBlock ]
{

/// Cube definition from Architect.
XData Cube [ XMLNamespace = "http://www.intersystems.com/deepsee" ]
{
<cube xmlns="http://www.intersystems.com/deepsee" name="waterPollutionCube" displayName="waterPollutionCube" disabled="false" abstract="false" sourceClass="dc.data.teccod.waterPollution" namedFactNums="true" countMeasureName="%COUNT" bucketSize="8" bitmapChunkInMemory="false" precompute="0" disableListingGroups="false" enableSqlRestrict="false">
  <dimension name="Country" disabled="false" hasAll="false" allCaption="All Country" allDisplayName="Country" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Country" disabled="false" sourceProperty="Country" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="2">
        <property name="CountryRegion" disabled="false" sourceExpression="%source.Country" hidden="false" isName="false" isDescription="false" isReference="false" useDisplayValue="false">
        </property>
        <property name="Uom" displayName="Uom" disabled="false" sourceExpression="%source.idwaterresultuom.Label" hidden="false" isName="false" isDescription="false" isReference="false" useDisplayValue="false">
        </property>
      </level>
      <level name="WaterBody_Name" displayName="WaterBody_Name" disabled="false" sourceProperty="idwatername.Label" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="5">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="idwaterbodycategories" disabled="false" hasAll="false" allCaption="All idwaterbodycategories" allDisplayName="idwaterbodycategories" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Label" disabled="false" sourceProperty="idwaterbodycategories.Label" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="3">
      </level>
      <level name="Notation" disabled="false" sourceProperty="idwaterbodycategories.Notation" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="4">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="idwatername" disabled="false" hasAll="false" allCaption="All idwatername" allDisplayName="idwatername" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Label" disabled="false" sourceProperty="idwatername.Label" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="5">
      </level>
      <level name="Notation" disabled="false" sourceProperty="idwatername.Notation" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="6">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="idwaterobservedproperty" disabled="false" hasAll="false" allCaption="All idwaterobservedproperty" allDisplayName="idwaterobservedproperty" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Label" disabled="false" sourceProperty="idwaterobservedproperty.Label" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="7">
      </level>
      <level name="Notation" disabled="false" sourceProperty="idwaterobservedproperty.Notation" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="8">
      </level>
      <level name="Definition" disabled="false" sourceProperty="idwaterobservedproperty.Definition" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="9">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="idwaterresultuom" disabled="false" hasAll="false" allCaption="All idwaterresultuom" allDisplayName="idwaterresultuom" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Label" disabled="false" sourceProperty="idwaterresultuom.Label" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="10">
      </level>
      <level name="Notation" disabled="false" sourceProperty="idwaterresultuom.Notation" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="11">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="phenomenonTimeReferenceYear" disabled="false" hasAll="false" allCaption="All phenomenonTimeReferenceYear" allDisplayName="phenomenonTimeReferenceYear" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="phenomenonTimeReferenceYear" disabled="false" sourceProperty="phenomenonTimeReferenceYear" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="12">
      </level>
    </hierarchy>
  </dimension>
  <measure name="resultMeanValue" disabled="false" sourceProperty="resultMeanValue" aggregate="AVG" type="number" hidden="false" searchable="false" factNumber="1">
  </measure>
  <measure name="TerraMarineProtected20162018" disabled="false" sourceProperty="TerraMarineProtected20162018" aggregate="AVG" type="number" hidden="false" searchable="false" factNumber="2">
  </measure>
  <measure name="droughtsfloodstemperature" disabled="false" sourceProperty="droughtsfloodstemperature" aggregate="AVG" type="number" hidden="false" searchable="false" factNumber="3">
  </measure>
  <measure name="VenueCount" disabled="false" sourceProperty="VenueCount" aggregate="SUM" type="number" hidden="false" searchable="false" factNumber="4">
  </measure>
</cube>
}

Parameter DOMAIN;

}
