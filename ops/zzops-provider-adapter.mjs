import { createProviderActivationPolicy, createProviderQualificationPolicy } from 'zzops/provider-owner';
export default {
  sourceReference: 'polyasr.owned-source', resultArtifact: 'owned-provider-result.json',
  workload() { throw new Error('Provider activation uses registered owner administration, not an inference train'); },
  verifyResult() { throw new Error('No inference qualification is asserted by provider activation'); },
  providerActivation: createProviderActivationPolicy({ app: 'polyasr', service: 'polyasr.service' }),
  providerQualification: createProviderQualificationPolicy({ app: 'polyasr', service: 'polyasr-qualification.service' }),
};
