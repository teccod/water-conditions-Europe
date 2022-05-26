/// 
Class dc.teccod.cube.waterPollutionCube Extends %DeepSee.CubeDefinition [ DependsOn = dc.data.teccod.waterPollution, ProcedureBlock ]
{

/// Cube definition from Architect.
XData Cube [ XMLNamespace = "http://www.intersystems.com/deepsee" ]
{
<cube xmlns="http://www.intersystems.com/deepsee" name="dc.teccod.cube.waterPollutionCube" displayName="dc.teccod.cube.waterPollutionCube" disabled="false" abstract="false" sourceClass="dc.data.teccod.waterPollution" namedFactNums="true" countMeasureName="%COUNT" bucketSize="8" bitmapChunkInMemory="false" precompute="0" disableListingGroups="false" enableSqlRestrict="false">
  <dimension name="Country" disabled="false" hasAll="false" allCaption="All Country" allDisplayName="Country" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="Country" disabled="false" sourceProperty="Country" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="2">
      </level>
    </hierarchy>
  </dimension>
  <dimension name="phenomenonTimeReferenceYear" disabled="false" hasAll="false" allCaption="All phenomenonTimeReferenceYear" allDisplayName="phenomenonTimeReferenceYear" type="data" hidden="false" showHierarchies="default">
    <hierarchy name="H1" disabled="false" hidden="false">
      <level name="phenomenonTimeReferenceYear" disabled="false" sourceProperty="phenomenonTimeReferenceYear" list="false" useDisplayValue="true" useAsFilter="true" hidden="false" factNumber="3">
      </level>
    </hierarchy>
  </dimension>
  <measure name="resultMeanValue" disabled="false" sourceProperty="resultMeanValue" aggregate="SUM" type="number" hidden="false" searchable="false" factNumber="1">
  </measure>
</cube>
}

Parameter DOMAIN;

}
